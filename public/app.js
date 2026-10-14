'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const form = $('prediction-form');
  const result = $('result-content');
  const emptyResult = result.innerHTML;
  const numericFields = [
    ['temperature', 'Temperature', -20, 120, 70, '°F', 0.1],
    ['humidity', 'Humidity', 0, 100, 50, '%', 1],
    ['pressure', 'Pressure', 28, 32, 30, 'inHg', 0.01],
    ['visibility', 'Visibility', 0, 20, 10, 'mi', 0.1],
    ['wind_speed', 'Wind speed', 0, 50, 10, 'mph', 0.1],
    ['precipitation', 'Precipitation', 0, 5, 0, 'in', 0.01],
    ['hour', 'Hour of day', 0, 23, 12, '0–23', 1],
    ['day_of_week_num', 'Day of week', 0, 6, 0, '0–6', 1],
    ['month', 'Month', 1, 12, 6, '1–12', 1]
  ];
  const infrastructure = ['amenity', 'bump', 'crossing', 'give_way', 'junction', 'no_exit', 'railway', 'roundabout', 'station', 'stop', 'traffic_calming', 'traffic_signal', 'turning_loop'];
  let predictionController = null;
  let requestVersion = 0;
  let hasResult = false;

  function fieldShell(id, label, unit = '') {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    const caption = document.createElement('label');
    caption.htmlFor = id;
    caption.textContent = label;
    if (unit) {
      const hint = document.createElement('span');
      hint.className = 'unit';
      hint.textContent = `(${unit})`;
      caption.append(hint);
    }
    wrap.append(caption);
    return wrap;
  }

  function numberField(spec, parent) {
    const [id, label, min, max, value, unit, step] = spec;
    const wrap = fieldShell(id, label, unit);
    const input = document.createElement('input');
    Object.assign(input, { id, name: id, type: 'number', min, max, step, value, defaultValue: String(value), required: true });
    wrap.append(input);
    $(parent).append(wrap);
  }

  function selectField(id, label, options, parent, defaultValue) {
    const wrap = fieldShell(id, label);
    const select = document.createElement('select');
    Object.assign(select, { id, name: id, required: true });
    options.forEach(value => select.add(new Option(value, value, value === defaultValue, value === defaultValue)));
    wrap.append(select);
    $(parent).append(wrap);
    return select;
  }

  numericFields.slice(0, 6).forEach(spec => numberField(spec, 'weather-fields'));
  selectField('wind_direction', 'Wind direction', ['Calm', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Variable'], 'weather-fields', 'Calm');
  selectField('weather_condition', 'Weather condition', ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog', 'Thunderstorm', 'Other'], 'weather-fields', 'Clear');
  selectField('state', 'State', ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'], 'location-fields', 'CA');
  const countyWrap = fieldShell('county', 'County');
  const county = document.createElement('input');
  Object.assign(county, { id: 'county', name: 'county', type: 'text', required: true, value: 'Los Angeles', defaultValue: 'Los Angeles', placeholder: 'Enter or select a county', maxLength: 150 });
  county.setAttribute('list', 'county-options');
  const countyOptions = document.createElement('datalist');
  countyOptions.id = 'county-options';
  countyWrap.append(county, countyOptions);
  $('location-fields').append(countyWrap);
  selectField('timezone', 'Timezone', ['US/Pacific', 'US/Mountain', 'US/Central', 'US/Eastern'], 'location-fields', 'US/Pacific');
  numericFields.slice(6).forEach(spec => numberField(spec, 'location-fields'));
  infrastructure.forEach(key => {
    const label = document.createElement('label');
    label.className = 'toggle-label';
    label.htmlFor = key;
    const title = key.replaceAll('_', ' ');
    label.append(document.createTextNode(title[0].toUpperCase() + title.slice(1)));
    const input = document.createElement('input');
    Object.assign(input, { type: 'checkbox', name: key, id: key });
    label.append(input);
    $('infrastructure-fields').append(label);
  });

  async function requestJSON(url, settings = {}) {
    const response = await fetch(url, settings);
    let data;
    try { data = await response.json(); }
    catch { throw new Error('The server returned an unreadable response. Please try again.'); }
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : typeof data.error === 'string' ? data.error : typeof data.message === 'string' ? data.message : `Request failed (${response.status}). Please try again.`;
      throw new Error(detail);
    }
    return data;
  }

  async function loadOptions() {
    const status = $('options-status');
    $('retry-options').hidden = true;
    status.textContent = 'Loading county options…';
    status.classList.remove('error');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const data = await requestJSON('/api/options', { signal: controller.signal });
      if (!Array.isArray(data.counties) || !data.counties.every(value => typeof value === 'string')) throw new Error('County options are not available. You can still enter a county manually.');
      countyOptions.replaceChildren(...data.counties.map(value => new Option(value, value)));
      status.textContent = `${data.counties.length.toLocaleString()} county options loaded. Select a county matching your state.`;
      if (data.metadata && typeof data.metadata === 'object') {
        $('model-metadata').textContent = JSON.stringify(data.metadata, null, 2);
        $('metadata-details').hidden = false;
      }
    } catch (error) {
      status.textContent = error.name === 'AbortError' ? 'County options timed out. Enter a county manually or retry.' : 'Could not load county options. Enter a county manually or retry.';
      status.classList.add('error');
      $('retry-options').hidden = false;
    } finally { clearTimeout(timeout); }
  }

  function clearResults() {
    hasResult = false;
    result.innerHTML = emptyResult;
    $('result-badge').textContent = 'AWAITING INPUT';
    $('form-error').hidden = true;
  }

  function setLoading(loading) {
    $('predict-button').disabled = loading;
    $('demo-button').disabled = loading;
    $('predict-button').innerHTML = loading ? 'Running model…' : 'Generate prediction <span aria-hidden="true">↗</span>';
    $('result-card').setAttribute('aria-busy', String(loading));
    Array.from(form.elements).forEach(input => {
      if (input.type !== 'reset') input.disabled = loading;
    });
  }

  function validPrediction(data) {
    return data && Number.isInteger(data.severity) && data.severity >= 1 && data.severity <= 4 &&
      typeof data.severity_name === 'string' && data.severity_name.trim() &&
      Number.isFinite(data.latitude) && Math.abs(data.latitude) <= 90 &&
      Number.isFinite(data.longitude) && Math.abs(data.longitude) <= 180 &&
      Array.isArray(data.probabilities) && data.probabilities.length === 4 &&
      data.probabilities.every(value => Number.isFinite(value) && value >= 0 && value <= 1);
  }

  function showPrediction(data) {
    const body = document.createElement('div');
    body.className = 'actual-result';
    body.innerHTML = '<div class="severity-label">Estimated traffic impact</div><div class="severity-output"><span class="severity-number"></span><div><h3></h3><p>Severity class · model estimate</p></div></div><h4 class="probability-title">Class probabilities</h4>';
    body.querySelector('.severity-number').textContent = data.severity;
    body.querySelector('h3').textContent = data.severity_name;
    data.probabilities.forEach((probability, index) => {
      const row = document.createElement('div');
      row.className = `probability-row${index + 1 === data.severity ? ' selected' : ''}`;
      row.innerHTML = '<div class="probability-label"><span></span><span></span></div><div class="probability-track"><div class="probability-fill"></div></div>';
      const labels = row.querySelectorAll('.probability-label span');
      labels[0].textContent = `Severity ${index + 1}`;
      labels[1].textContent = `${(probability * 100).toFixed(1)}%`;
      row.querySelector('.probability-fill').style.width = `${probability * 100}%`;
      body.append(row);
    });
    const coordinates = document.createElement('dl');
    coordinates.className = 'coordinates';
    ['latitude', 'longitude'].forEach(key => {
      const pair = document.createElement('div');
      const label = document.createElement('dt');
      label.textContent = `Estimated ${key}`;
      const value = document.createElement('dd');
      value.textContent = data[key].toFixed(5);
      pair.append(label, value);
      coordinates.append(pair);
    });
    const note = document.createElement('p');
    note.className = 'result-note';
    note.textContent = 'Returned by the model API for your submitted scenario. Not a verified incident location.';
    body.append(coordinates, note);
    result.replaceChildren(body);
    $('result-badge').textContent = 'MODEL OUTPUT';
    hasResult = true;
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (predictionController || !form.reportValidity()) return;
    if (!county.value.trim()) { county.setCustomValidity('Please enter a county.'); county.reportValidity(); return; }
    const payload = {};
    new FormData(form).forEach((value, key) => { payload[key] = String(value).trim(); });
    numericFields.forEach(([key]) => { payload[key] = Number(payload[key]); });
    infrastructure.forEach(key => { payload[key] = $(key).checked; });
    const version = ++requestVersion;
    const controller = new AbortController();
    predictionController = controller;
    const timeout = setTimeout(() => controller.abort(), 45000);
    clearResults();
    setLoading(true);
    $('result-badge').textContent = 'PROCESSING';
    result.innerHTML = '<div class="loading-result"><span class="spinner" aria-hidden="true"></span>Analyzing your scenario…</div>';
    try {
      const data = await requestJSON('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
      if (version !== requestVersion) return;
      if (!validPrediction(data)) throw new Error('The model returned incomplete or invalid results. No prediction has been displayed.');
      showPrediction(data);
    } catch (error) {
      if (version !== requestVersion) return;
      clearResults();
      $('result-badge').textContent = 'UNAVAILABLE';
      $('form-error').textContent = error.name === 'AbortError' ? 'The prediction timed out. Please try again.' : `Unable to generate prediction. ${error.message}`;
      $('form-error').hidden = false;
    } finally {
      clearTimeout(timeout);
      if (version === requestVersion) { predictionController = null; setLoading(false); }
    }
  });
  county.addEventListener('input', () => county.setCustomValidity(''));
  form.addEventListener('input', () => { if (hasResult) $('result-badge').textContent = 'INPUTS CHANGED'; });
  form.addEventListener('change', () => { if (hasResult) $('result-badge').textContent = 'INPUTS CHANGED'; });
  form.addEventListener('reset', () => {
    ++requestVersion;
    if (predictionController) predictionController.abort();
    predictionController = null;
    setLoading(false);
    county.setCustomValidity('');
    clearResults();
  });
  $('demo-button').addEventListener('click', () => {
    form.reset();
    const preset = { temperature: 62, humidity: 82, pressure: 29.8, visibility: 4, wind_speed: 15, precipitation: 0.12, wind_direction: 'SW', weather_condition: 'Rain', state: 'CA', county: 'Los Angeles', timezone: 'US/Pacific', hour: 17, day_of_week_num: 2, month: 11, crossing: true, junction: true, traffic_signal: true };
    Object.entries(preset).forEach(([key, value]) => { if (typeof value === 'boolean') $(key).checked = value; else $(key).value = value; });
    $('options-status').textContent = 'Demo scenario populated. Generate a prediction when you are ready.';
  });
  $('retry-options').addEventListener('click', loadOptions);
  function updateNavigation() {
    const current = location.hash || '#predictor';
    document.querySelectorAll('.nav-link').forEach(link => {
      const active = link.getAttribute('href') === current;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
  }
  window.addEventListener('hashchange', updateNavigation);
  updateNavigation();
  loadOptions();
})();

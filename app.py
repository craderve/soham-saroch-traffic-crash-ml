"""TrafficLens: the original trained models behind a small Vercel-ready API."""
from functools import lru_cache
import json
import math
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "streamlit_deployment"
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
NUMBERS = {
    "temperature": (-20, 120, 70), "humidity": (0, 100, 50),
    "pressure": (28, 32, 30), "visibility": (0, 20, 10),
    "wind_speed": (0, 50, 10), "precipitation": (0, 5, 0),
    "hour": (0, 23, 12), "day_of_week_num": (0, 6, 0), "month": (1, 12, 6),
}
CHOICES = {
    "state": ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"],
    "timezone": ["US/Pacific", "US/Mountain", "US/Central", "US/Eastern"],
    "wind_direction": ["Calm", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "Variable"],
    "weather_condition": ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm", "Other"],
}
INFRASTRUCTURE = {
    "amenity": "Amenity", "bump": "Bump", "crossing": "Crossing", "give_way": "Give_Way",
    "junction": "Junction", "no_exit": "No_Exit", "railway": "Railway",
    "roundabout": "Roundabout", "station": "Station", "stop": "Stop",
    "traffic_calming": "Traffic_Calming", "traffic_signal": "Traffic_Signal", "turning_loop": "Turning_Loop",
}
SEVERITIES = {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Severe"}


@lru_cache(maxsize=1)
def counties():
    return sorted(set(line.strip() for line in (MODELS / "unique_counties.txt").read_text().splitlines() if line.strip()))


@lru_cache(maxsize=1)
def load_models():
    # Load only the trusted artifacts shipped with the source project, never user files.
    import joblib
    severity = joblib.load(MODELS / "severity_classifier.pkl")
    location = joblib.load(MODELS / "location_regressor.pkl")
    preprocessor = joblib.load(MODELS / "preprocessor.pkl")
    severity.set_params(n_jobs=1)
    location.set_params(n_jobs=1)
    preprocessor.set_params(n_jobs=1)
    for estimator in getattr(location, "estimators_", []):
        estimator.set_params(n_jobs=1)
    return severity, location, preprocessor


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object containing the scenario inputs.")
    values = {}
    for key, (low, high, default) in NUMBERS.items():
        raw = payload.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be a number between {low} and {high}.")
        try:
            value = float(raw)
        except (ValueError, TypeError):
            raise ValueError(f"{key} must be a number between {low} and {high}.") from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        if key in ("hour", "day_of_week_num", "month"):
            if not value.is_integer():
                raise ValueError(f"{key} must be a whole number.")
            value = int(value)
        values[key] = value
    for key, allowed in CHOICES.items():
        value = payload.get(key, allowed[0])
        if value not in allowed:
            raise ValueError(f"Choose a valid {key}.")
        values[key] = value
    values["county"] = payload.get("county", "Los Angeles")
    if values["county"] not in counties():
        raise ValueError("Choose a county from the supplied county list.")
    for key in INFRASTRUCTURE:
        value = payload.get(key, False)
        if type(value) is not bool:
            raise ValueError(f"{key} must be true or false.")
        values[key] = value
    return values


def input_frame(values):
    import pandas as pd
    # Retain the source app's preprocessing assumptions for model compatibility.
    # Twilight is fixed to Day and wind chill is estimated, not observed.
    row = {
        "Temperature(F)": values["temperature"], "Wind_Chill(F)": values["temperature"] - 5,
        "Humidity(%)": values["humidity"], "Pressure(in)": values["pressure"],
        "Visibility(mi)": values["visibility"], "Wind_Speed(mph)": values["wind_speed"],
        "Precipitation(in)": values["precipitation"], "County": values["county"],
        "State": values["state"], "Timezone": values["timezone"],
        "Wind_Direction": values["wind_direction"], "Weather_Condition": values["weather_condition"],
        "Sunrise_Sunset": "Day", "Civil_Twilight": "Day", "Nautical_Twilight": "Day", "Astronomical_Twilight": "Day",
        "Hour": values["hour"], "DayOfWeek": values["day_of_week_num"], "Month": values["month"],
        "IsWeekend": int(values["day_of_week_num"] >= 5),
    }
    row.update({column: int(values[key]) for key, column in INFRASTRUCTURE.items()})
    return pd.DataFrame([row])


@app.get("/")
def index():
    return send_from_directory(ROOT / "public", "index.html")


@app.get("/styles.css")
@app.get("/app.js")
def asset():
    return send_from_directory(ROOT / "public", request.path.lstrip("/"))


@app.get("/api/health")
def health():
    return jsonify(status="ok", project="Soham Saroch TrafficLens")


@app.get("/api/options")
def options():
    return jsonify(counties=counties(), metadata=json.loads((MODELS / "model_metadata.json").read_text()))


@app.post("/api/predict")
def predict():
    try:
        values = validate(request.get_json(silent=True))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    try:
        severity_model, location_model, preprocessor = load_models()
        processed = preprocessor.transform(input_frame(values))
        raw_severity = int(severity_model.predict(processed)[0])
        # The original XGBoost classifier uses zero-based class labels.
        severity = raw_severity + 1 if hasattr(severity_model, "objective") else raw_severity
        coordinates = location_model.predict(processed)[0]
        probabilities = [float(value) for value in severity_model.predict_proba(processed)[0]]
        latitude, longitude = (float(value) for value in coordinates)
        if severity not in SEVERITIES or not all(math.isfinite(value) for value in [latitude, longitude, *probabilities]):
            raise RuntimeError("Invalid model output")
        return jsonify(severity=severity, severity_name=SEVERITIES[severity], latitude=latitude,
                       longitude=longitude, probabilities=probabilities)
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="The trained model could not complete this prediction. Please try again later."), 503


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="The scenario request is too large."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

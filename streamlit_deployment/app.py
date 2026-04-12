import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="US Traffic Crash Machine Learning",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
SEVERITY_COLORS = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
SEVERITY_NAMES = {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Severe"}

DAY_MAPPING = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

DEFAULT_COUNTIES = [
    "Los Angeles", "Cook", "Harris", "Maricopa", "San Diego", "Orange", 
    "Miami-Dade", "Kings", "Queens", "Dallas", "Riverside", "San Bernardino",
    "Clark", "Tarrant", "Santa Clara", "Broward", "New York", "Wayne",
    "Bexar", "King", "Bronx", "Suffolk", "Alameda", "Nassau", "Palm Beach",
    "Hillsborough", "Pinellas", "Middlesex", "Sacramento", "Philadelphia",
    "Contra Costa", "Oakland", "Fulton", "Cuyahoga", "Fresno", "Westchester",
    "Fairfax", "Duval", "DeKalb", "Kern", "Ventura", "Gwinnett", "Orange County",
    "Mecklenburg", "Collin", "Jefferson", "Lee", "Pima", "Wake", "Travis"
]

WIND_DIRECTIONS = ["Calm", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "Variable"]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm", "Other"]
STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
TIMEZONES = ["US/Pacific", "US/Mountain", "US/Central", "US/Eastern"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Load models and metadata
@st.cache_resource
def load_models() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Dict], Optional[Any]]:
    """Load trained models and preprocessor"""
    try:
        # Load metadata
        with open('streamlit_deployment/model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load models
        severity_model = joblib.load('streamlit_deployment/severity_classifier.pkl')
        location_model = joblib.load('streamlit_deployment/location_regressor.pkl')
        preprocessor = joblib.load('streamlit_deployment/preprocessor.pkl')
        
        # Load feature names if available
        feature_names = None
        if os.path.exists('streamlit_deployment/feature_names.pkl'):
            with open('streamlit_deployment/feature_names.pkl', 'rb') as f:
                feature_names = joblib.load(f)
        
        return severity_model, location_model, preprocessor, metadata, feature_names
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

@st.cache_data
def load_counties() -> List[str]:
    """Load counties list from file"""
    try:
        with open('unique_counties.txt', 'r', encoding='utf-8') as f:
            counties = [line.strip() for line in f if line.strip()]
        return sorted(counties)
    except FileNotFoundError:
        return sorted(DEFAULT_COUNTIES)
    except Exception as e:
        st.error(f"Error loading counties: {e}")
        return ["Los Angeles"]  # Default fallback

def render_sidebar(metadata: Dict) -> None:
    """Render sidebar with model information"""
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"**Creation Date:** {metadata['creation_date'][:10]}")
        
        if 'classification' in metadata['models_info']:
            clf_info = metadata['models_info']['classification']
            st.success(f"**Severity Model:** {clf_info['model_name']}")
            if 'classification' in metadata['performance_metrics']:
                acc = metadata['performance_metrics']['classification']['test_accuracy']
                st.metric("Test Accuracy", f"{acc:.4f}")
        
        if 'regression' in metadata['models_info']:
            reg_info = metadata['models_info']['regression']
            st.success(f"**Location Model:** {reg_info['model_name']}")
            if 'regression' in metadata['performance_metrics']:
                r2 = metadata['performance_metrics']['regression']['r2']
                st.metric("R² Score", f"{r2:.4f}")
    
        # Add spacer to push footer to bottom
        st.markdown("<div style='margin-top: auto;'></div>", unsafe_allow_html=True)
        
        # Footer with developer info
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: left; color: #666; font-size: 14px; margin-top: auto;'>
                <p>
                    👨‍💻 <strong>Developer:</strong> Anıl Metin<br>
                    🔗 <strong>GitHub:</strong> <a href='https://github.com/anilmetin0' target='_blank' style='color: #1f77b4; text-decoration: none;'>github.com/anilmetin0</a><br>
                    🚗 <strong>Project:</strong> US Traffic Crash Machine Learning
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

def render_weather_inputs() -> Dict:
    """Render weather condition inputs"""
    st.subheader("🌤️ Weather Conditions")
    return {
        'temperature': st.slider("Temperature (°F)", -20, 120, 70),
        'humidity': st.slider("Humidity (%)", 0, 100, 50),
        'pressure': st.slider("Pressure (in)", 28.0, 32.0, 30.0, 0.1),
        'visibility': st.slider("Visibility (mi)", 0.0, 20.0, 10.0, 0.5),
        'wind_speed': st.slider("Wind Speed (mph)", 0, 50, 10),
        'precipitation': st.slider("Precipitation (in)", 0.0, 5.0, 0.0, 0.1),
        'wind_direction': st.selectbox("Wind Direction", WIND_DIRECTIONS),
        'weather_condition': st.selectbox("Weather Condition", WEATHER_CONDITIONS)
    }

def render_location_time_inputs(counties_list: List[str]) -> Dict:
    """Render location and time inputs"""
    st.subheader("📍 Location & Time")
    
    state = st.selectbox("State", STATES)
    county = st.selectbox("County", counties_list, index=0)
    timezone = st.selectbox("Timezone", TIMEZONES)
    
    # Time features
    hour = st.slider("Hour of Day", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", DAYS_OF_WEEK)
    month = st.slider("Month", 1, 12, 6)
    
    # Convert day of week to number
    day_of_week_num = DAY_MAPPING[day_of_week]
    is_weekend = 1 if day_of_week_num >= 5 else 0
    
    return {
        'state': state,
        'county': county,
        'timezone': timezone,
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_week_num': day_of_week_num,
        'month': month,
        'is_weekend': is_weekend
    }

def render_infrastructure_inputs() -> Dict:
    """Render road infrastructure inputs"""
    st.subheader("🛣️ Road Infrastructure")
    return {
        'amenity': st.checkbox("Near Amenity"),
        'bump': st.checkbox("Speed Bump"),
        'crossing': st.checkbox("Crossing"),
        'give_way': st.checkbox("Give Way"),
        'junction': st.checkbox("Junction"),
        'no_exit': st.checkbox("No Exit"),
        'railway': st.checkbox("Railway"),
        'roundabout': st.checkbox("Roundabout"),
        'station': st.checkbox("Station"),
        'stop': st.checkbox("Stop Sign"),
        'traffic_calming': st.checkbox("Traffic Calming"),
        'traffic_signal': st.checkbox("Traffic Signal"),
        'turning_loop': st.checkbox("Turning Loop")
    }

def create_input_dataframe(weather_data: Dict, location_data: Dict, infrastructure_data: Dict) -> pd.DataFrame:
    """Create input DataFrame from form data"""
    return pd.DataFrame({
        'Temperature(F)': [weather_data['temperature']],
        'Wind_Chill(F)': [weather_data['temperature'] - 5],  # Approximation
        'Humidity(%)': [weather_data['humidity']],
        'Pressure(in)': [weather_data['pressure']],
        'Visibility(mi)': [weather_data['visibility']],
        'Wind_Speed(mph)': [weather_data['wind_speed']],
        'Precipitation(in)': [weather_data['precipitation']],
        'County': [location_data['county']],
        'State': [location_data['state']],
        'Timezone': [location_data['timezone']],
        'Wind_Direction': [weather_data['wind_direction']],
        'Weather_Condition': [weather_data['weather_condition']],
        'Sunrise_Sunset': ['Day'],  # Simplified
        'Civil_Twilight': ['Day'],
        'Nautical_Twilight': ['Day'],
        'Astronomical_Twilight': ['Day'],
        'Amenity': [int(infrastructure_data['amenity'])],
        'Bump': [int(infrastructure_data['bump'])],
        'Crossing': [int(infrastructure_data['crossing'])],
        'Give_Way': [int(infrastructure_data['give_way'])],
        'Junction': [int(infrastructure_data['junction'])],
        'No_Exit': [int(infrastructure_data['no_exit'])],
        'Railway': [int(infrastructure_data['railway'])],
        'Roundabout': [int(infrastructure_data['roundabout'])],
        'Station': [int(infrastructure_data['station'])],
        'Stop': [int(infrastructure_data['stop'])],
        'Traffic_Calming': [int(infrastructure_data['traffic_calming'])],
        'Traffic_Signal': [int(infrastructure_data['traffic_signal'])],
        'Turning_Loop': [int(infrastructure_data['turning_loop'])],
        'Hour': [location_data['hour']],
        'DayOfWeek': [location_data['day_of_week_num']],
        'Month': [location_data['month']],
        'IsWeekend': [location_data['is_weekend']]
    })

def make_predictions(input_data: pd.DataFrame, severity_model: Any, location_model: Any, preprocessor: Any) -> Tuple[int, np.ndarray]:
    """Make predictions using the models"""
    # Preprocess data
    x_processed = preprocessor.transform(input_data)
    
    # Make predictions
    severity_pred = severity_model.predict(x_processed)[0]
    location_pred = location_model.predict(x_processed)[0]
    
    # Adjust prediction for XGBoost if needed
    if hasattr(severity_model, 'objective'):
        severity_pred = severity_pred + 1
    
    return severity_pred, location_pred

def display_prediction_results(severity_pred: int, location_pred: np.ndarray) -> None:
    """Display prediction results"""
    st.markdown("---")
    st.header("📋 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Crash Severity")
        st.metric(
            "Predicted Severity", 
            f"{SEVERITY_COLORS.get(severity_pred, '⚪')} Level {severity_pred} - {SEVERITY_NAMES.get(severity_pred, 'Unknown')}"
        )
    
    with col2:
        st.subheader("📍 Predicted Location")
        st.metric("Latitude", f"{location_pred[0]:.6f}")
        st.metric("Longitude", f"{location_pred[1]:.6f}")
    
    # Risk assessment
    st.subheader("⚠️ Risk Assessment")
    if severity_pred <= 2:
        st.success("Low to Moderate Risk - Exercise normal caution")
    elif severity_pred == 3:
        st.warning("High Risk - Exercise extra caution")
    else:
        st.error("Very High Risk - Consider alternative routes if possible")

# Load models
severity_model, location_model, preprocessor, metadata, feature_names = load_models()

# Load counties list
counties_list = load_counties()

def main():
    """Main application function"""
    st.title("🚗 US Traffic Crash Machine Learning")
    st.markdown("---")
    
    if severity_model is None:
        st.error("Failed to load models. Please check model files.")
        return
    
    # Render sidebar
    render_sidebar(metadata)
    
    # Main prediction interface
    st.header("🔮 Make Predictions")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            weather_data = render_weather_inputs()
        
        with col2:
            location_data = render_location_time_inputs(counties_list)
        
        with col3:
            infrastructure_data = render_infrastructure_inputs()
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)
        
        if submitted:
            try:
                # Create input data
                input_data = create_input_dataframe(weather_data, location_data, infrastructure_data)
                
                # Make predictions
                severity_pred, location_pred = make_predictions(input_data, severity_model, location_model, preprocessor)
                
                # Display results
                display_prediction_results(severity_pred, location_pred)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()

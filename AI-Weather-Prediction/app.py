
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI Weather Dashboard",
    page_icon="🌦️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "weather_model.pkl"
DATA_PATH = BASE_DIR / "weather.csv"

st.title("🌦️ AI Weather Condition Prediction")
st.write("Machine learning based weather classification dashboard.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Enter Weather Details")

temperature = st.sidebar.slider(
    "Temperature (°C)", -10, 50, 28
)
humidity = st.sidebar.slider(
    "Humidity (%)", 0, 100, 65
)
wind_speed = st.sidebar.slider(
    "Wind Speed (km/h)", 0, 100, 15
)
pressure = st.sidebar.slider(
    "Pressure (hPa)", 950, 1050, 1010
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", f"{temperature} °C")
col2.metric("Humidity", f"{humidity}%")
col3.metric("Wind Speed", f"{wind_speed} km/h")
col4.metric("Pressure", f"{pressure} hPa")

input_data = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "pressure": pressure
}])

st.divider()
st.subheader("🤖 AI Weather Prediction")

if st.button("Predict Weather", type="primary"):
    prediction = str(model.predict(input_data)[0])

    icons = {
        "Sunny": "☀️",
        "Rainy": "🌧️",
        "Cloudy": "☁️",
        "Foggy": "🌫️"
    }

    st.success(
        f"{icons.get(prediction, '🌤️')} Predicted condition: {prediction}"
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        probability_df = pd.DataFrame({
            "Weather": model.classes_,
            "Estimated Probability (%)": probabilities * 100
        }).sort_values(
            "Estimated Probability (%)", ascending=False
        )

        st.subheader("Prediction Probability")
        st.bar_chart(
            probability_df.set_index("Weather")
        )
        st.dataframe(probability_df, use_container_width=True)

st.divider()
st.subheader("📊 Weather Dataset Analysis")

if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)

    left, right = st.columns(2)

    with left:
        st.write("Weather condition distribution")
        st.bar_chart(data["weather"].value_counts())

    with right:
        st.write("Temperature distribution")
        st.bar_chart(data["temperature"])

    with st.expander("View weather dataset"):
        st.dataframe(data, use_container_width=True)

st.caption(
    "Educational demonstration only. Predictions are based on "
    "the training dataset and are not live weather forecasts."
)


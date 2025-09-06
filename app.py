import streamlit as st
import numpy as np
import pandas as pd
import os
import pydeck as pdk
from model import preprocess_data, train_models

st.title("Illegal Fishing Prediction (Single Input)")

# Preload and train on the full data at app startup
@st.cache_resource
def load_and_train():
    df = pd.read_csv('dataset/ais_data.csv')
    processed_df, X_scaled, scaler, le = preprocess_data(df)
    iso_forest, autoencoder, X_train = train_models(X_scaled)
    return scaler, le, iso_forest

scaler, le, iso_forest = load_and_train()

st.markdown("### Enter AIS features for one vessel:")

with st.form("ais_input_form"):
    sog = st.number_input("Speed Over Ground (knots)", min_value=0.0, max_value=30.0, value=5.0)
    cog = st.number_input("Course Over Ground (degrees)", min_value=0.0, max_value=360.0, value=90.0)
    heading = st.number_input("Heading (degrees)", min_value=0.0, max_value=360.0, value=90.0)
    speed_change = st.number_input("Speed Change (knots)", min_value=-10.0, max_value=10.0, value=0.0)
    heading_change = st.number_input("Heading Change (degrees)", min_value=-180.0, max_value=180.0, value=0.0)
    nav_status_options = list(le.classes_)
    nav_status = st.selectbox("Navigational Status", nav_status_options, index=0)
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.9)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.2)
    submitted = st.form_submit_button("Predict")

if submitted:
    nav_encoded = le.transform([nav_status])[0]
    X_single = np.array([[sog, cog, heading, speed_change, heading_change, nav_encoded]])
    X_single_scaled = scaler.transform(X_single)
    iso_pred = iso_forest.predict(X_single_scaled)
    is_anomaly = iso_pred[0] == -1

    # Prediction message
    st.markdown("### Prediction Result")
    result_label = "Suspicious (Illegal Fishing)" if is_anomaly else "Normal"
    if is_anomaly:
        st.error("Suspicious activity detected (possible illegal fishing)!")
    else:
        st.success("Normal activity detected.")

    # Map display with pydeck for clearer marker
    st.markdown("### Vessel Location on Map")

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=10,
        pitch=0
    )

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=pd.DataFrame({'lat': [latitude], 'lon': [longitude]}),
        get_position='[lon, lat]',
        get_radius=500,
        get_fill_color=[255, 0, 0, 160],
        pickable=True
    )

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[layer]
    )

    st.pydeck_chart(deck)

    # Record the input and result into a log CSV using pd.concat instead of deprecated append
    new_row = {
        "sog": sog,
        "cog": cog,
        "heading": heading,
        "speed_change": speed_change,
        "heading_change": heading_change,
        "navigationalstatus": nav_status,
        "latitude": latitude,
        "longitude": longitude,
        "model_prediction": result_label
    }

    log_path = "prediction_log.csv"
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([new_row])
    log_df.to_csv(log_path, index=False)

    st.info("Prediction, location, and input recorded into prediction_log.csv.")

# --------------------------
# Display Popular Ports Section

st.markdown("---")
st.header("Popular Ports and Locations")

ports_data = {
    "Port": ["Mumbai Port", "Chennai Port", "Kolkata Port", "Kochi Port", "Visakhapatnam Port"],
    "Latitude": [18.9388, 13.0827, 22.5726, 9.9312, 17.6868],
    "Longitude": [72.8355, 80.2707, 88.3639, 76.2673, 83.2185]
}

ports_df = pd.DataFrame(ports_data)

st.dataframe(ports_df)

st.markdown("### Popular Ports Map")

view_state_ports = pdk.ViewState(
    latitude=20.0,
    longitude=80.0,
    zoom=4,
    pitch=0,
)

layer_ports = pdk.Layer(
    "ScatterplotLayer",
    data=ports_df.rename(columns={"Latitude":"lat", "Longitude":"lon"}),
    get_position='[lon, lat]',
    get_radius=80000,
    get_fill_color=[0, 128, 255, 160],
    pickable=True,
)

r = pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state_ports,
    layers=[layer_ports],
)

st.pydeck_chart(r)

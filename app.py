import streamlit as st
import pandas as pd
from model import preprocess_data, train_models, predict_anomalies
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Illegal Fishing Detection with AIS Data")

uploaded_file = st.file_uploader("Upload your AIS CSV dataset", type=["csv"])

# Use default dataset path if no file uploaded
if uploaded_file is None:
    try:
        st.info("Using default dataset at dataset/ais_data.csv")
        df = pd.read_csv('dataset/ais_data.csv')
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        st.stop()
else:
    df = pd.read_csv(uploaded_file)

st.write("Dataset preview:")
st.dataframe(df.head())

try:
    # Preprocess data
    processed_df, X_scaled, scaler, le = preprocess_data(df)
    st.success("Data preprocessing completed.")

    # Train models
    iso_forest, autoencoder, X_train = train_models(X_scaled)
    st.success("Models trained successfully.")

    # Predict anomalies
    results_df, iso_preds, mse, threshold = predict_anomalies(processed_df, X_scaled, iso_forest, autoencoder, X_train)

    # Show anomaly counts
    n_anomalies = results_df['combined_anomaly'].sum()
    st.write(f"Detected {n_anomalies} suspicious vessel movements.")

    # Display some anomaly examples
    st.write("Sample suspicious detections:")
    st.dataframe(results_df[results_df['combined_anomaly'] == 1][['mmsi', 'sog', 'cog', 'heading', 'iso_anomaly', 'dl_anomaly']].head(10))

    # If ground truth labels exist (optional)
    if 'label' in results_df.columns:
        acc = accuracy_score(results_df['label'], results_df['combined_anomaly'])
        st.write(f"Model accuracy on provided labels: {acc:.2f}")

        cm = confusion_matrix(results_df['label'], results_df['combined_anomaly'])
        st.write("Confusion Matrix:")
        st.write(cm)

except Exception as e:
    st.error(f"Error processing the data: {e}")

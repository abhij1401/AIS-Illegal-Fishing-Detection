import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def preprocess_data(df):
    df['mmsi'] = df['mmsi'].astype(str)
    df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
    df['cog'] = pd.to_numeric(df['cog'], errors='coerce')
    df['heading'] = pd.to_numeric(df['heading'], errors='coerce')
    df['navigationalstatus'] = df['navigationalstatus'].fillna('Unknown')

    # Filter fishing vessels only
    df = df[df['shiptype'].str.lower() == 'fishing'].copy()
    df.dropna(subset=['sog', 'cog', 'heading'], inplace=True)

    # Feature engineering
    df.sort_values(['mmsi'], inplace=True)
    df['speed_change'] = df.groupby('mmsi')['sog'].diff().fillna(0)
    df['heading_change'] = df.groupby('mmsi')['heading'].diff().fillna(0)

    # Encode navigation status
    le = LabelEncoder()
    df['navigation_encoded'] = le.fit_transform(df['navigationalstatus'])

    features = ['sog', 'cog', 'heading', 'speed_change', 'heading_change', 'navigation_encoded']
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler, le

def train_models(X_scaled):
    # Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X_scaled)

    # Autoencoder model
    input_dim = X_scaled.shape[1]
    encoding_dim = 3

    model = Sequential([
        Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(input_dim, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train-test split for autoencoder
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    model.fit(X_train, X_train,
              epochs=50,
              batch_size=64,
              shuffle=True,
              validation_data=(X_test, X_test),
              verbose=0)

    return iso_forest, model, X_train

def predict_anomalies(df, X_scaled, iso_forest, autoencoder, X_train):

    # Isolation Forest anomalies
    iso_preds = iso_forest.predict(X_scaled)
    df['iso_anomaly'] = np.where(iso_preds == -1, 1, 0)

    # Autoencoder reconstruction error
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

    threshold = np.percentile(np.mean(np.power(X_train - autoencoder.predict(X_train), 2), axis=1), 95)
    df['dl_anomaly'] = (mse > threshold).astype(int)

    # Combined anomalies
    df['combined_anomaly'] = ((df['iso_anomaly'] == 1) | (df['dl_anomaly'] == 1)).astype(int)

    return df, iso_preds, mse, threshold
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('dataset/ais_data.csv')

    # Preprocess
    processed_df, X_scaled, _, _ = preprocess_data(df)

    # Train models
    iso_forest, autoencoder, X_train = train_models(X_scaled)

    # Predict
    results_df, _, _, _ = predict_anomalies(processed_df, X_scaled, iso_forest, autoencoder, X_train)

    print(f"Detected {results_df['combined_anomaly'].sum()} anomalies.")


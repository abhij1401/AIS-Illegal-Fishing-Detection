🎣 AIS-Illegal-Fishing-Detection 🚨
An open-source tool that uses AIS vessel data and machine learning to detect illegal fishing activities in real time.
Features an interactive Streamlit interface for easy input, anomaly detection, and geospatial visualization.

✨ Features
🤖 Hybrid ML Models: Isolation Forest & Autoencoder for robust anomaly detection

🖥️ Interactive Web App: User-friendly Streamlit interface for instant predictions

📍 Map Visualization: View vessel locations and suspicious activity on maps

📝 Activity Logging: Logs all user inputs and predictions automatically

⚓ Popular Ports: Displays key maritime ports with coordinates on an interactive map

🚀 Installation
Clone the repo:

bash
git clone https://github.com/abhij1401/AIS-Illegal-Fishing-Detection.git
cd AIS-Illegal-Fishing-Detection
Create and activate Conda environment (recommended):

bash
conda create -n fishing_detection python=3.9 -y
conda activate fishing_detection
Install dependencies:

bash
pip install -r requirements.txt
🏃‍♂️ How to Use
Run the app:

bash
streamlit run app.py
Enter vessel movement and navigational features.

Get real-time detection of suspicious (potential illegal) fishing activity.

See vessel position on interactive maps alongside popular ports.

Your inputs and prediction results are saved for audit purposes.

🗂️ Project Structure
text
├── dataset/             # AIS data files (not bundled)  
├── venv/                # Virtual environment folder (optional)  
├── app.py               # Streamlit app interface  
├── model.py             # Data preprocessing and ML model code  
├── prediction_log.csv   # Logs of predictions and inputs  
├── requirements.txt     # Python package dependencies  
└── README.md            # Project overview  
⚙️ Model Summary
Preprocessing: Cleans and prepares AIS data, features like speed & heading changes

Anomaly Detection: Isolation Forest flags unusual movement patterns

Autoencoder: Detects deviations from learned normal vessel behaviors

🤝 Contributing
Contributions welcome! Submit issues, feature requests, or pull requests.

📄 License
This project is licensed under the MIT License.

📫 Contact
Questions, suggestions, or collaborations? Reach out to the maintainer.

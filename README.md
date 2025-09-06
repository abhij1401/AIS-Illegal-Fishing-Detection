text
# 🎣 AIS-Illegal-Fishing-Detection 🚨

An open-source tool that uses AIS vessel data and machine learning to detect **illegal fishing activities** in real time.  
Includes an interactive Streamlit interface for data input, anomaly detection, and geospatial visualization.

---

## ✨ Features

- 🤖 **Hybrid ML Models:** Isolation Forest & Autoencoder for anomaly detection  
- 🖥️ **Streamlit Web App:** Friendly interface for instant predictions  
- 🗺️ **Map Visualizations:** View vessel locations and suspicious activity  
- 📝 **Activity Logging:** Logs all user inputs and predictions automatically  
- ⚓ **Popular Ports:** Displays key maritime ports on an interactive map  

---

## 🛠️ Installation

git clone https://github.com/abhij1401/AIS-Illegal-Fishing-Detection.git
cd AIS-Illegal-Fishing-Detection
conda create -n fishing_detection python=3.9 -y
conda activate fishing_detection
pip install -r requirements.txt

text

---

## ▶️ How to Use

streamlit run app.py

text

1. Enter vessel movement and navigational features.  
2. Get instant detection of suspicious (illegal) fishing activity.  
3. See vessel position on interactive map alongside popular ports.  
4. Your inputs and prediction results are saved for audit.

---

## 📂 Project Structure

dataset/ # AIS data files (not bundled)
venv/ # Virtual environment (optional)
app.py # Streamlit app interface
model.py # Data preprocessing and ML model code
requirements.txt # Python dependencies
prediction_log.csv # Logs of predictions
README.md # Project overview
LICENSE # MIT License

text

---

## 📈 Model Summary

- **Preprocessing:** Cleans and organizes AIS data; creates features like speed & heading changes  
- **Anomaly Detection:** Isolation Forest flags unusual vessel movement patterns  
- **Autoencoder:** Detects deviations from learned normal vessel behaviors  

---

## 📄 License

MIT License © 2025 Abhishek Jadhav

---

## 📬 Contact

Questions or contributions? Reach out to the maintainer.

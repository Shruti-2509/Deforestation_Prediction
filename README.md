# Forest Guard — Deforestation Prediction Dashboard

A Streamlit-based environmental monitoring app for analyzing deforestation patterns, predicting forest risk from tabular data, scanning satellite images, and interacting with a forest protection chatbot.

## 🚀 Features

- **Overview & Analytics**: Visualize historical deforestation trends, forest cover, and risk distribution.
- **Tabular Predictor**: Enter land parameters and predict deforestation risk using a trained Random Forest classifier.
- **Image Classification**: Upload satellite images and detect forest vs. deforested zones from pixel analysis.
- **Chatbot Assistant**: Ask about forest protection, fire prevention, illegal logging, reforestation, and sustainable land use.
- **Offline-friendly**: Works locally with the included dataset and does not require external APIs to run, although optional OpenRouter integration is supported.

## 📁 Project Structure

- `gui.py` — Main Streamlit application.
- `data/data.csv` — Dataset used for analytics and training.
- `requirements.txt` — Python dependencies.
- `RUN_DASHBOARD.bat` — One-click launcher for Windows.
- `Deforestation_Prediction (2).ipynb` — Notebook for experiments and model exploration.
- `images/` — Supporting visualization assets.
- `OFFLINE_GUIDE.md` — Instructions for offline execution.

## ⚙️ Requirements

- Python 3.9+
- Install dependencies from `requirements.txt`.

## 💻 Setup

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## ▶️ Run the App

From the project root, launch the dashboard with:

```powershell
streamlit run gui.py
```

Or double-click `RUN_DASHBOARD.bat` on Windows for an automated launch.

## 🌐 Optional OpenRouter Chatbot

The chatbot can use OpenRouter if you provide an API key in an environment variable named `OPENROUTER_API_KEY`.

Create a `.env` file in the project root with:

```ini
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=deepseek/deepseek-chat
```

If no key is found, the app falls back to a built-in rule-based assistant.

## 📌 Notes

- Ensure `data/data.csv` is present before running the app.
- The model training is performed in-app from the dataset on startup.
- The image classification module performs a simple color-based scan and is intended for demonstration.

## 🧠 Contact

Use this README as the main developer reference for installing and launching the Forest Guard dashboard.

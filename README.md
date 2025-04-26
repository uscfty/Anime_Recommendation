# Anime Recommendation System

This project is deployed online and can be accessed directly:

🌐 **Live Demo:** [Click to Open](https://anime-recommendation-futianyu0715.onrender.com)

🎥 **Demo Video:** [Watch here]()
---

## Features

- Flask-based REST API
- Pre-trained scikit-learn model
- Deployed on Render.com

## Technologies

- Python 3.11
- Flask 3.1
- scikit-learn 1.2.2
- Hosted via Render

---





# Anime Recommendation System API

This project provides a Flask-based REST API for serving an anime recommendation model using `scikit-learn`.

---

## 📦 Environment Setup

### ✅ Python Version

- **Python 3.11.x** (Recommended)
- Compatible with macOS, Linux, Windows

> ⚠️ Avoid using Python 3.12+ due to known issues with some packages (e.g., `numpy`, `scikit-learn`, and missing `distutils`)

---

## 🔧 Setup Instructions

1. **Open the folder into VSC IDE**

```bash
cd Anime_Recommendation_test_cleanest
```
2. **Environment Preparation**
# Exit any currently activated virtual environment (if applicable):
```bash
deactivate
rm -rf .venv
#Remove all dependencies, interpreter copies, package caches, etc. that you installed
```
# Create virtual environment
```bash
python3.11 -m venv .venv
```
3. **Activate the environment**
# macOS/Linux
```bash
source .venv/bin/activate

# Windows (cmd)
.venv\Scripts\activate
```
4. ** Dependency Versions**
# Install the dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
5. ** Dependency Versions**
```txt
Flask==3.1.0
flask_cors==5.0.1
joblib==1.2.0
numpy==1.25.0
pandas==2.0.3
Requests==2.32.3
scikit-learn==1.2.2
```
6. ** Running the API Server**
```bash
python server.py
```

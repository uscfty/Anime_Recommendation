# Anime_Recommendation
Recommendation Animates to Users
# Anime Recommendation System - Local Test Guide

# Anime Recommendation System API

This project provides a Flask-based REST API for serving an anime recommendation model trained with `scikit-learn`.

---

## 🔧 Environment Setup

### Python Version

- Python 3.11.8 (recommended)
- Created using: `/Users/futianyu/anaconda3/bin/python3.11`

### Required Packages

The required packages are listed in `requirements.txt`:
Flask==3.1.0
flask_cors==5.0.1
joblib==1.2.0
numpy==1.25.0
pandas==2.0.3
Requests==2.32.3
scikit-learn==1.2.2



> ⚠️ **Note:** The model was originally trained using `scikit-learn 1.3.0`.  
> You may see a warning when loading the model with `1.2.2`, but functionality should remain unaffected unless custom objects are used.  
> See: [Model Persistence Compatibility](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations)

---

## 💻 How to Run

1. **Clone this project**  
   (or ensure you're inside the project root directory)

2. **Create virtual environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py

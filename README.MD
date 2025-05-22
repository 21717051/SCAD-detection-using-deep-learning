# SCAD Detection Using CNN - Web Application

This project uses a Convolutional Neural Network (CNN) to detect **Spontaneous Coronary Artery Dissection (SCAD)** in angiographic images. It includes a simple web interface built with Flask, allowing users to upload images and receive predictions (SCAD or Normal).

---

## 📁 Project Structure

```
SCAD_CNN_WebApp/
├── app.py                  # Flask web application
├── train.py                # CNN model training script
├── predict.py              # Prediction logic
├── static/uploads/         # Folder to store uploaded images
├── templates/index.html    # Web UI HTML template
├── models/cnn_model.h5     # Trained CNN model
├── requirements.txt        # Project dependencies
└── README.md               # Project instructions and overview
```

---

## ⚙️ Requirements

- Python 3.8+
- pip
- All required packages listed in `requirements.txt`

To install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Navigate to the project folder:
```bash
cd SCAD_CNN_WebApp
```

2. Launch the Flask app:
```bash
python app.py
```

3. Open your browser and go to:
```
http://127.0.0.1:5000
```

4. Upload an angiographic image and click **Detect SCAD**.  
   The model will predict and display whether the image is **SCAD** or **Normal**.

---

## 🧠 Model Overview

- A custom CNN architecture was trained on a dataset of labeled angiogram images.
- The model includes convolutional, pooling, and dense layers optimized for binary classification.
- The prediction logic is wrapped inside a Flask web app for easy testing.
- Uploaded images are preprocessed and classified on the backend.

---

## 📊 Model Performance

- **Accuracy**: 1.0000 
- **validation Accuracy**: 0.9995  
- **Training Loss**: 3.6482e-04 
- **Validation Loss**: 0.0395

Performance was evaluated using a hold-out test set and confusion matrix analysis.

---

## 🔄 Future Work

- Integrate YOLOv8 for region-based SCAD detection
- Add class activation maps (CAM) or Grad-CAM visualization
- Improve UI design and support batch uploads
- Deploy as a standalone web service using Docker or Streamlit

---

## 🗃️ Key Folders Explained

| Folder | Purpose |
|--------|---------|
| `models/` | Stores trained CNN model (`cnn_model.h5`) |
| `static/uploads/` | Stores uploaded images temporarily |
| `templates/` | HTML templates for the web interface |
| `train.py` | Model training logic |
| `predict.py` | Handles prediction from image input |

---

## 👩‍💻 Author

**Urva Patel**  
COMP6016 – Master of Computing  
Curtin University – May 2025

---

# 👁️ Computer Vision Interactive Lab

An interactive Computer Vision web application built using
Python, OpenCV, DeepFace and Streamlit.

The application allows users to upload images, select different
computer vision techniques, execute them interactively, and
understand the results and calculations.

---

## 🚀 Features

### 🎯 Template Matching

Uses OpenCV template matching to locate a reference image
inside an input image.

Technique:

`cv2.matchTemplate()`

Method:

`TM_CCOEFF_NORMED`

---

### 👤 Viola–Jones Face Detection

Detects faces using the classical Viola–Jones algorithm.

Pipeline:

Haar Features  
↓  
Integral Image  
↓  
AdaBoost  
↓  
Cascade Classifier  
↓  
Face Detection

---

### 🧠 DeepFace + ArcFace

Performs deep-learning-based face verification.

Pipeline:

Input Image  
↓  
Face Detection  
↓  
Face Alignment  
↓  
ArcFace Embedding  
↓  
Cosine Distance  
↓  
Threshold Comparison  
↓  
Same / Different Person

---

### 🔵 DeepFace + FaceNet

Uses FaceNet embeddings to compare faces.

The system calculates the distance between two facial
embeddings and determines whether the images represent
the same person.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pandas
- Pillow
- DeepFace
- TensorFlow
- TF-Keras
- ArcFace
- FaceNet
- Viola–Jones
- Template Matching

---

## 📂 Project Structure

```text
Computer-Vision-Interactive-Lab/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
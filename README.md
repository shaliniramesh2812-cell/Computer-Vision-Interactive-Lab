# 🔍 Interactive Computer Vision Analysis Web App

An interactive web application for exploring and comparing multiple **Computer Vision techniques** through a simple and user-friendly interface.

The application allows users to upload images, select different Computer Vision methods, execute the selected algorithms, visualize the results, and understand the calculations and decision-making behind each method.

🌐 **Live Demo:**  
[Launch the Computer Vision Web App]
(https://computer-vision-interactive-lab.streamlit.app/)



---

## 📌 Project Overview

Computer Vision involves extracting meaningful information from images using image processing, pattern recognition, and machine learning techniques.

This project provides an interactive platform where users can experiment with different Computer Vision algorithms without writing code.

The application currently supports:

- 🎯 Template Matching
- 👤 Viola–Jones Face Detection
- 🧠 DeepFace Face Verification
- 🔎 FaceNet Face Verification
- 📊 Performance and result comparison
- 📖 Algorithm explanations
- 🧮 Similarity/distance calculations
- 🖼️ Visual result visualization

The main objective is to make Computer Vision algorithms easier to understand by combining **implementation, visualization, calculations, and explanations** in one web application.

---

## ✨ Key Features

### 1. 🎯 Template Matching

Template Matching is used to locate a smaller image or pattern inside a larger image.

The application:

- Accepts an input image.
- Accepts a template image.
- Compares the template with different regions of the input image.
- Calculates a matching score.
- Identifies the best matching location.
- Displays the detected region visually.

### 2. 👤 Viola–Jones Face Detection

The application implements the classical **Viola–Jones object detection framework** using OpenCV's Haar Cascade classifier.

Features include:

- Face detection from uploaded images.
- Bounding boxes around detected faces.
- Configurable detection sensitivity.
- Minimum face-size filtering.
- False-positive reduction.
- Single-person detection mode.
- Number of detected faces.

The implementation also handles image orientation to improve detection reliability.

### 3. 🧠 DeepFace Face Verification

DeepFace is used for deep-learning-based face verification.

The application compares:

- Input image
- Reference face image

It calculates a facial embedding distance and determines whether the two faces are sufficiently similar according to the selected model and threshold.

The result includes:

- Verification status
- Facial distance
- Verification threshold
- Comparison result

### 4. 🔎 FaceNet Face Verification

FaceNet is another deep-learning approach for face recognition and verification.

The application generates facial representations and compares the embeddings between the input and reference images.

The result provides:

- Face similarity information
- Embedding distance
- Verification decision
- Model-based comparison

---

## 🏗️ Application Workflow

```text
                 ┌─────────────────────┐
                 │     Upload Image     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Select CV Methods   │
                 └──────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
   Template Matching   Viola–Jones       Face Verification
                                          │
                                  ┌───────┴────────┐
                                  │                │
                                  ▼                ▼
                              DeepFace          FaceNet
                                  │                │
                                  └───────┬────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │  Results & Metrics  │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Visualization +     │
                              │ Explanation         │
                              └─────────────────────┘
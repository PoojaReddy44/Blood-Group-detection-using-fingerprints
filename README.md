# Blood-Group-detection-using fingerprint
This repository contains the complete source code, datasets, and results related to the project "Blood Group Detection Using Fingerprints". The project explores the innovative application of deep learning models (VGG16-based CNN) to classify a person’s blood group directly from fingerprint images — eliminating the need for traditional invasive blood sampling methods.
The system processes fingerprint images and predicts the corresponding blood group class using a trained model.

# 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Getting Started](#getting-started)
4. [Dataset](#dataset)
5. [Models](#models)
6. [Results](#results)
7. [Sample Dataset](#sample-dataset)
8. [Usage](#usage)
9. [Model Download Instructions](#model-download-instructions)
10. [Requirements](#requirements)
11. [Contributing](#contributing)
12. [License](#license)

# Project Overview
This project, Blood Group Detection Using Fingerprints, aims to automate the process of determining a person's blood group by analyzing their fingerprint images using deep learning techniques. Traditional methods for blood group detection require collecting blood samples through invasive procedures, which can be time-consuming, costly, and sometimes uncomfortable for patients.
By leveraging machine learning models such as VGG16 and image processing techniques, this project offers a non-invasive, quick, and cost-effective alternative to conventional blood sample testing. Users can upload fingerprint images through a simple web interface built with Flask (Python), and the system predicts the blood group with high confidence, displaying both the result and the confidence score.

The system includes:

A user-friendly web application (login, signup, home, prediction pages),

Pre-trained deep learning models for prediction,

Sample datasets of fingerprint images categorized by blood group,

Visualizations like model accuracy, loss graphs, and architecture diagram,

A demo video and project presentation for better understanding.

# Folder Structure 
## 📁 Folder Structure

```plaintext
blood/                         # Main project folder
│
├── app.py                     # Flask main application file
├── enhance_fingerprint.py     # Script for fingerprint enhancement
├── train_model.py             # Script to train the blood group detection model
│
├── Model/                     # Contains the trained model and label files
│   ├── keras_model.h5         # Trained Keras model (not pushed to GitHub, too large)
│   └── labels.txt             # Label mappings for prediction
│
├── dataset_blood_group/       # Dataset containing fingerprint images categorized by blood group
│   ├── A+/                   # Images for blood group A+
│   ├── A-/                   # Images for blood group A-
│   ├── B+/                   # Images for blood group B+
│   ├── B-/                   # Images for blood group B-
│   ├── AB+/                  # Images for blood group AB+
│   ├── AB-/                  # Images for blood group AB-
│   ├── O+/                   # Images for blood group O+
│   └── O-/                   # Images for blood group O-
│
├── static/                   # Static files (CSS, Images, Video, PDF)
│   ├── style.css             # Custom stylesheet
│   ├── lab_image.png         # Image shown on Home page
│   ├── architecture.png      # Model architecture visualization
│   ├── accuracy.png          # Accuracy graph
│   ├── loss.png              # Loss graph
│   ├── loss_ratio.png        # Loss ratio graph
│   ├── project_demo.mp4      # Project demo video
│   ├── BloodGroup_Fingerprint_Presentation.pdf # About/Project presentation PDF
│   └── .DS_Store             # macOS system file (ignored)
│
├── templates/                # Flask HTML templates (Jinja2)
│   ├── about.html            # About page with embedded PDF
│   ├── accuracy.html         # Accuracy video page
│   ├── form.html             # User details form (name, age, gender)
│   ├── home.html             # Home page (with Get Started & Watch Video)
│   ├── index.html            # Main fingerprint upload and prediction page
│   ├── login.html            # Login page
│   ├── signup.html           # Sign-up page for new users
│   ├── upload.html           # Upload page for fingerprint image
│   ├── watch_video.html      # Video playback page
│   └── welcome.html          # Welcome screen with auto-redirect
│
├── .gitignore                # Git ignored files (e.g., model, venv)
├── .gitattributes            # LFS tracked files settings
├── requirements.txt          # Required Python packages list
└── README.md                 # Project description file (you are reading this!)

```
# Getting Started
## Prerequisites
Make sure the following are installed:

Python 3.10+

pip (Python package manager)

Git

Virtual Environment (recommended)
## Installation steps:
Here’s your **Installation Steps** in **perfect, ready-to-copy format for README.md** — properly marked and clean:

---

## 🛠️ Installation Steps

### 1. **Clone the repository:**

```bash
git clone https://github.com/PoojaReddy44/Blood-Group-detection-using-fingerprints.git
cd Blood-Group-detection-using-fingerprints
```

---

### 2. **(Optional) Create and activate a virtual environment:**

For **macOS/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

For **Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. **Install required packages:**

```bash
pip install -r requirements.txt
```

---

### 4. **Download the model file (`keras_model.h5`) and labels (`labels.txt`):**

⚠️ **Note:**
The trained model file (`keras_model.h5`) is large and not included in this repository due to GitHub file size limits.

👉 \[Provide your Google Drive / External Download Link Here]

After downloading, place the following files into the `Model/` directory:

```
Model/
 ├── keras_model.h5
 └── labels.txt
```

---

### 5. **Run the Flask app:**

```bash
python app.py
```

---

### 6. **Open your browser and visit:**

```bash
http://127.0.0.1:5000
```

---

### 7. **(Optional) Retrain the model:**

```bash
python train_model.py
```

---





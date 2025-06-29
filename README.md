# Blood-Group-detection-using fingerprint
This repository contains the complete source code, datasets, and results related to the project "Blood Group Detection Using Fingerprints". The project explores the innovative application of deep learning models (VGG16-based CNN) to classify a person’s blood group directly from fingerprint images — eliminating the need for traditional invasive blood sampling methods.
The system processes fingerprint images and predicts the corresponding blood group class using a trained model.

# 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Getting Started](#getting-started)
4. [Dataset](#dataset)
5. [Models](#-models)
6. [Results](#results)
7. [Sample Dataset](#-sample-dataset)
8. [Usage](#usage)
9. [Model Download Instructions](#-model-download-instructions)
10. [Requirements](#-requirements)
11. [License](#-license)

## 🎥 Demo Video

[[Watch the Demo Video]](https://drive.google.com/file/d/1luOW0r8AjcxoFeOK19psw-xvvxsY4USN/view?usp=sharing)


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

### 4.Model Generation Instructions:**

> ⚠️ **Note:** The `Model/` directory (which contains the trained model `keras_model.h5` and `labels.txt`) is **not included** in this repository due to file size restrictions.

You can generate the model by training it yourself using the following command:

```bash
python train_model.py
```

This will:

* Train the CNN model (e.g., VGG16) on the dataset
* Automatically create the `Model/` folder
* Save the trained model as `keras_model.h5` and the label mapping as `labels.txt`

Once training is complete, the `Model/` directory will be ready for use during prediction.



⚠️ **Note: For Reference**
The trained model file (`keras_model.h5`) is large and not included in this repository due to GitHub file size limits.

👉 [Download Keras_model.h5](https://drive.google.com/file/d/1-qlmxs8OlovCTEYW5GTco-f6IKZP1B76/view?usp=sharing)

[Download Labels.txt](https://drive.google.com/file/d/1T6d8DYMi7AsL8MS3GPcr9gKMMn9aiKZt/view?usp=sharing)



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
# DataSet
The dataset contains fingerprint images organized into folders by blood group type. There are approximately 6,000–7,000 images spread across the eight blood group categories.

Each subfolder within the dataset_blood_group/ directory represents a specific blood group and contains corresponding fingerprint images labeled accordingly,

Ensure that you have sufficient storage space before downloading or expanding the dataset.

The dataset is essential for training the machine learning models to classify fingerprint patterns into the respective blood groups.

All images are processed and resized to a uniform size of 224x224 pixels before feeding into the models like VGG16 for prediction.

# 🧠 Models 

This project explores the following CNN architecture:

* **VGG16**

All model-related files are located under the `Model/` directory, with:

* A trained model file: `keras_model.h5`
* A label mapping file: `labels.txt`

**Note:** The model was trained and tested via Python scripts (`train_model.py`).

# Results
Performance metrics and training graphs are stored in the static/ folder for easy access and visualization.

* **Graphs**:
Includes the following visualizations to represent the model's performance:

* **accuracy.png**: Training and validation accuracy plot

* **loss.png**: Training and validation loss plot

* **loss_ratio.png**: Loss ratio over epochs

* **architecture.png**: Visualization of the model architecture

These graphs help in understanding the model’s learning progress and efficiency during training.

# 📂 Sample Dataset
The dataset_blood_group/ folder includes fingerprint images categorized into subfolders according to blood group types.

Each subfolder (e.g., A+/, B-/, O+/, etc.) contains sample fingerprint images that demonstrate the structure and naming convention of the full dataset.

This allows users to understand the expected input format and organization before running or training the model.

---
# Usage

**Dataset Preparation:**
Ensure that the `dataset_blood_group/` folder is populated with the required fingerprint images categorized by blood groups.

**Running the Model:**
Execute the `train_model.py` script to train the model. The script is designed to:

* Load the dataset
* Train the model on the training dataset
* Evaluate the model on the test dataset

**Viewing Results:**
After training, check the `static/` folder for visualizations like accuracy, loss, and model architecture graphs.

**Testing on Sample Data:**
You can test the model using images from the `dataset_blood_group/` or by using additional external datasets collected for validation purposes.

# 📥 Model Download Instructions

The trained model file (`keras_model.h5`) is **not included** in this repository due to GitHub's file size restrictions.

👉 You can generate the model by training it yourself using the following command:

```bash
python train_model.py
```


**For reference:[🔗 Google Drive Model Link](https://drive.google.com/file/d/1-qlmxs8OlovCTEYW5GTco-f6IKZP1B76/view?usp=sharing)**  

After downloading:

1. Place the following files in the `Model/` directory of this project:

   * `keras_model.h5` — Trained Keras Model
   * `labels.txt` — Class labels file

### Example:

```
blood/
│
├── Model/
│   ├── keras_model.h5    # Place downloaded model here
│   └── labels.txt        # Place labels file here
```

⚠️ **Note:** Without these files, the prediction functionality in the app will not work.

# 📝 Requirements

Before running the project, make sure the following Python packages are installed:

```
Flask
tensorflow
Pillow
numpy
```

### You can install all required packages by running:

```bash
pip install -r requirements.txt
```

### Sample `requirements.txt` file content:

```
Flask==3.0.0
tensorflow==2.14.0
Pillow==10.0.0
numpy==1.24.3
```

### Optional (for development):

```
JupyterLab  # if you want to run training or data exploration notebooks
```

# 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
You are free to use, modify, and distribute this project with proper attribution.





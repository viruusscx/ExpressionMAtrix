# Emotion Detection Web App

## 1. Overview
This project is a **deep learning-based emotion detection system** built from scratch using a **Convolutional Neural Network (CNN)**.  
It can classify human emotions (e.g., Happy, Sad, Angry, Surprise, Neutral) from facial images.  
The model is deployed via a **Flask web application**, making it accessible through a browser interface.

---

## 2. Significance
- **Emotion AI**: Used in healthcare, customer experience, and social robotics.
- **Practical Learning**: Covers dataset preprocessing, CNN design, training, evaluation, and deployment.
- **End-to-End Development**: From model training to a working web app.

---

## 3. Project Structure


emotion-detection-web/
│
├── app.py                # Flask application (runs the web server)
├── emotion_model.h5      # Trained CNN model for emotion detection
├── static/               # Static assets (CSS, JavaScript, images)
├── templates/            # HTML templates for web pages
├── requirements.txt      # Python dependencies
└── trainingmodel.ipynb   # Jupyter Notebook used for training the model



---

## 4. How We Trained the Model (Deep Explanation)

### 4.1 Dataset Preparation
- **Dataset**: Facial images labeled with emotions.
- **Preprocessing**:
  - Images converted to grayscale (reduces complexity from 3 channels to 1).
  - Resized to **48×48 pixels** for uniformity.
  - Normalized pixel values (0–1) for faster training.

### 4.2 CNN Architecture Design
A CNN was chosen because:
- Convolutional layers capture spatial patterns in images.
- Pooling layers reduce dimensionality and prevent overfitting.
- Fully connected layers handle final classification.

**Model Layers**:
1. **Input Layer**: Accepts 48×48 grayscale images.
2. **Convolutional Layers**: Extract features using filters.
3. **ReLU Activation**: Adds non-linearity to learn complex patterns.
4. **MaxPooling Layers**: Downsamples feature maps, reducing computational cost.
5. **Dropout Layers**: Prevent overfitting by randomly disabling neurons during training.
6. **Flatten Layer**: Converts 2D feature maps to a 1D vector.
7. **Dense Layers**: Perform classification.
8. **Softmax Output Layer**: Produces probabilities for each emotion class.

### 4.3 Training Process
- **Loss Function**: `categorical_crossentropy` (multi-class classification).
- **Optimizer**: `Adam` (adaptive learning rate for efficient convergence).
- **Batch Size & Epochs**: Model trained in small batches for multiple epochs.
- **Validation**: Dataset split into training and validation sets to track performance.
- **Checkpointing**: Best model saved as `emotion_model.h5`.

---

## 5. Web Application Flow
1. User uploads an image (or webcam frame can be added in future).
2. Flask reads the image and preprocesses it (resize, grayscale, normalize).
3. The trained CNN (`emotion_model.h5`) predicts emotion probabilities.
4. Prediction is displayed on the web page in real time.

---

## 6. Installation & Usage

### 6.1 Prerequisites
- Python 3.8+
- pip

### 6.2 Setup
```bash
pip install -r requirements.txt

6.3 Run Application

python app.py

6.4 Open in Browser
http://127.0.0.1:5000/


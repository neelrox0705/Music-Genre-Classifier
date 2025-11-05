# 🎵 Music Genre Classification

## 📘 Overview
This project focuses on classifying music tracks into different genres using **Machine Learning**.  
By extracting various **audio features** from songs such as MFCCs, Chroma, and Spectral properties using **Librosa**, we train multiple ML models to predict the genre of unseen tracks.

The project is part of the **Machine Learning course (Mid-Sem Submission)** at **BITS Pilani**.

---

## 🧠 Objective
To build and evaluate a machine learning model that can automatically identify the **genre** of a song based on its **audio characteristics**.

---

## 📂 Repository Contents

| File | Description |
|------|--------------|
| `Untitled3.ipynb` | Feature extraction and preprocessing |
| `Untitled4.ipynb` | Model training, testing, and evaluation |
| `scaler.pkl` | Saved `StandardScaler` used for feature normalization |
| `svm_genre_model.pkl` | Trained SVM model for music genre classification |

---

## ⚙️ Technologies Used
- **Language:** Python  
- **Libraries:**  
  - `librosa` – Audio feature extraction  
  - `numpy`, `pandas` – Data handling and manipulation  
  - `scikit-learn` – ML models, scaling, evaluation metrics  
  - `matplotlib`, `seaborn` – Data visualization  

---

## 🎶 Dataset
The project uses the **GTZAN Music Genre Dataset**, containing 10 genres:

> 🎸 Blues, 🎻 Classical, 🤠 Country, 💃 Disco, 🎧 Hiphop, 🎷 Jazz, 🤘 Metal, 🎤 Pop, 🌴 Reggae, 🎸 Rock

- Each genre includes **100 tracks** of **30 seconds** each.  
- Dataset path used in the notebook:
  ```
  C:/Users/neelj/OneDrive/Desktop/ML/archive/Data/genres_original
  ```
Change it for running locally
---

## 🧩 Workflow

### 1. Feature Extraction
Extracted a rich set of audio features from each track:
- **MFCCs**
- **Chroma**
- **Spectral Centroid**
- **Spectral Bandwidth**
- **Spectral Rolloff**
- **Zero Crossing Rate**
- **RMS Energy**
- **Mel Spectrogram**
- **Tempo**

Each song is represented by the mean and standard deviation of these features.

---

### 2. Data Preprocessing
- Encoded genre labels using `LabelEncoder`
- Normalized features using `StandardScaler`
- Split dataset into **training and testing sets**

---

### 3. Model Training
Trained and compared the following models:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**

Saved the trained **SVM model** as the best-performing one.

---

### 4. Evaluation
- Evaluated using **Accuracy**, **Classification Report**, and **Confusion Matrix**
- Visualized performance using heatmaps and plots

**Best Model:** `SVM`  
**Accuracy:** *(to be updated after final testing)*

---

## 💾 Saved Models
| File | Purpose |
|------|----------|
| `svm_genre_model.pkl` | Trained SVM classifier |
| `scaler.pkl` | Scaler used for feature normalization |

These files can be directly loaded for inference without retraining.

---

## 🚀 Future Work
- Implement a **Flask or Streamlit web app** for real-time genre prediction  
- Add **deep learning models** (e.g., CNNs on spectrogram images)  
- Perform **hyperparameter tuning** for higher accuracy  
- Explore additional audio features and datasets  

---

## 👥 Team Members
- Neel Patel
- Samar Chaudhary
- Krishna Dadhich


---

## 📅 Course Details
- **Course:** Machine Learning  
- **Institute:** BITS Pilani  
- **Project Stage:** Mid-Semester Submission  

---

### 🏁 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in order:
   - `Untitled3.ipynb` → Feature extraction
   - `Untitled4.ipynb` → Model training and evaluation

4. Use the saved `.pkl` files for inference.

---

### 📜 License
This project is for **academic and educational purposes** only.

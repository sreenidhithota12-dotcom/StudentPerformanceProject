# 🎓 Student Performance Predictor using Machine Learning

A machine learning project that predicts student exam scores based on academic, personal, and environmental factors. The project includes data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## 📌 Project Overview

This project applies supervised machine learning techniques to analyze the factors affecting student performance and predict examination scores.

The complete workflow includes:

- Data preprocessing
- Missing value handling
- Feature encoding
- Feature scaling
- Principal Component Analysis (PCA)
- Model training and evaluation
- Deployment using Streamlit

---
## ✨ Key Features

- Interactive Streamlit web application
- End-to-end machine learning pipeline
- Data preprocessing and feature engineering
- Multiple regression and classification models
- Ensemble learning techniques
- Clustering analysis
- Feature importance visualization
- Real-time exam score prediction

## 📂 Dataset

**Dataset:** Student Performance Factors Dataset

The dataset contains features related to:

- Hours Studied
- Attendance
- Previous Scores
- Sleep Hours
- Tutoring Sessions
- Physical Activity
- Family Income
- Internet Access
- Teacher Quality
- Motivation Level
- School Type
- Peer Influence
- Distance from Home
- Gender
- and several other academic and demographic attributes.

---

## 🚀 Machine Learning Pipeline

### Data Preprocessing

- Removed unnecessary columns
- Handled missing values
- One-Hot Encoding for categorical variables
- Feature Scaling using StandardScaler

### Feature Engineering

- Principal Component Analysis (PCA)
- Train / Validation / Test Split
- 5-Fold Cross Validation

### Regression Models

- Linear Regression
- Lasso Regression

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification Models

- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Evaluation Metrics:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

### Ensemble Learning

- Random Forest
- XGBoost
- Voting Classifier (Hard Voting)
- Voting Classifier (Soft Voting)

### Clustering

- K-Means Clustering
- Agglomerative Clustering

Evaluation Metrics:

- Silhouette Score
- Davies-Bouldin Index

### Neural Network

- Multi-Layer Perceptron (MLP)

---

## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Pickle (Model Serialization)
---

## 🌐 Streamlit Application

The project includes a Streamlit web application where users can:

- Enter student details
- Predict examination score
- View the model's most influential features for the prediction.
---

## 📁 Project Structure

```
StudentPerformanceProject/
│
├── app/
│   └── app.py
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── notebooks/
│   └── student_performance_ml.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/sreenidhithota12-dotcom/StudentPerformanceProject.git
```

Move into the project directory

```bash
cd StudentPerformanceProject
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
python -m streamlit run app/app.py
```

## 📊 Sample Prediction

The application predicts the student's expected examination score based on multiple academic and personal factors.

Example:

```
Predicted Exam Score: 69.62
```

The application also displays the most influential features contributing to the prediction.

---
## 📈 Results

- Successfully trained and evaluated multiple machine learning models.
- Compared regression, classification, clustering, and ensemble techniques.
- Developed a Streamlit application for real-time exam score prediction.
- Identified the most influential features affecting predicted student performance.

## 🔮 Future Improvements

- SHAP-based model explainability
- Hyperparameter tuning
- Additional regression models
- Improved interactive dashboard
- Model comparison visualization

---

## 👨‍💻 Author

**Thota Sreenidhi**

- GitHub: https://github.com/sreenidhithota12-dotcom
- LinkedIn: https://www.linkedin.com/in/sreenidhi-thota-3080a2336

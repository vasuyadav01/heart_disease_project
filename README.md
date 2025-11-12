# 🫀 Heart Disease Classification

This project demonstrates an end-to-end machine learning workflow to **predict the likelihood of heart disease** in patients based on various medical attributes. It leverages popular Python data-science libraries for data exploration, preprocessing, model training, evaluation, and interpretation.

---

## 📋 Project Overview

The notebook walks through the complete ML lifecycle:

1. **Problem Definition** — Predict whether a person has heart disease.
2. **Data Collection** — Use the *Cleveland Heart Disease Dataset* from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) or [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
3. **Evaluation Metric** — Target performance of ≥95% accuracy.
4. **Feature Analysis** — Understand each feature’s relationship to heart disease.
5. **Model Building** — Train and test multiple ML algorithms.
6. **Experimentation** — Compare models, tune hyperparameters, and finalize the best one.

---

## 🧰 Tech Stack

| Category             | Libraries / Tools   |
| -------------------- | ------------------- |
| **Language**         | Python              |
| **Data Handling**    | pandas, numpy       |
| **Visualization**    | matplotlib, seaborn |
| **Machine Learning** | scikit-learn        |
| **Environment**      | Jupyter Notebook    |

---

## 🧠 Key Steps

### 1. Data Loading & Exploration

* Load dataset and inspect shape, missing values, and data types.
* Perform **exploratory data analysis (EDA)** to find patterns and correlations.

### 2. Data Preprocessing

* Handle missing values and encode categorical variables.
* Split dataset into training and testing sets.
* Apply scaling and transformations if needed.

### 3. Model Training

* Train models such as:

  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Random Forest
  * Support Vector Machine (SVM)

### 4. Model Evaluation

* Evaluate models using **accuracy**, **precision**, **recall**, and **F1-score**.
* Visualize performance using confusion matrices.

### 5. Hyperparameter Tuning

* Use GridSearchCV or RandomizedSearchCV to optimize parameters.
* Select the best model based on performance metrics.

### 6. Final Model & Insights

* Save the trained model for future use.
* Summarize findings and feature importances.

---

## 📊 Example Outputs

* Correlation heatmaps between features.
* Heart disease frequency plots by gender and age.
* Accuracy comparison between ML algorithms.
* Final performance metrics of the best model.

---

## 🧩 Dataset Information

**Source:**

* [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
* [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Target Variable:**
`target` → 1 (disease present) / 0 (no disease)

**Sample Features:**

| Feature  | Description                       |
| -------- | --------------------------------- |
| age      | Age in years                      |
| sex      | 1 = male, 0 = female              |
| cp       | Chest pain type                   |
| trestbps | Resting blood pressure            |
| chol     | Serum cholesterol (mg/dl)         |
| fbs      | Fasting blood sugar > 120 mg/dl   |
| thalach  | Maximum heart rate achieved       |
| exang    | Exercise induced angina           |
| oldpeak  | ST depression induced by exercise |

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/heart-disease-classification.git
   cd heart-disease-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook end-to-end-heart-disease-classification.ipynb
   ```
4. Run all cells to reproduce results.

---

## 📦 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📈 Results Summary

* Achieved **~95% accuracy** on the test dataset.
* Random Forest and Logistic Regression performed best overall.
* Features like *chest pain type*, *max heart rate*, and *ST depression* were strong predictors.

---

## 🧾 License

This project is released under the [MIT License](LICENSE).

---

## ✨ Acknowledgements

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
* [Kaggle Heart Disease Dataset](https://www.kaggle.com/)
* Open-source Python community

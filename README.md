# 🧠 Machine Learning Basics – End-to-End Practice Repository

Welcome to my **Machine Learning Basics** repository!  
This project is a **comprehensive collection of Jupyter Notebooks, datasets, and saved models** created during my ML learning journey.  
It covers everything from **data preprocessing and Python libraries practice** to the implementation of **core machine learning algorithms** with real datasets.

---

## 📌 About the Project

The purpose of this repository is to **practice and understand machine learning concepts step by step**.  
I explored:
- Data preprocessing (cleaning, encoding, scaling, splitting)
- ML algorithms (Linear Regression, KNN, SVM, Decision Tree, Random Forest, ANN)
- Model saving/loading with `pickle`
- Visualization using Matplotlib
- Experiments with real-world datasets like **Iris, Insurance, Student marks, Sports, and Disease prediction**.

This repository can serve as a **learning guide for beginners** who want to see implementations of ML basics in action.

---

## 📂 Project Structure

### 🔧 **1. Data Preprocessing**
- Notebook: `DataPreprocessing.ipynb`  
- Concepts covered:
  - Handling missing values  
  - Encoding categorical variables  
  - Feature scaling (StandardScaler, MinMaxScaler)  
  - Train-test split  

---

### 📚 **2. Python Libraries Practice**
- `numpyPractice.ipynb` → Array creation, reshaping, indexing, mathematical operations.  
- `pandasPractice.ipynb` → DataFrames, filtering, grouping, merging, handling missing values.  
- `matplotlibPractice.ipynb` → Basic plotting, bar charts, line graphs, scatter plots, histograms.  

---

### 🤖 **3. Machine Learning Algorithms**

#### 📈 Linear Regression
- Notebooks: 
  - `LinearRegression Implementation.ipynb`  
  - `Student Marks Predictor.ipynb`  
- Concepts: Predicting continuous values, regression line, Mean Squared Error.  

#### 👥 K-Nearest Neighbors (KNN)
- Notebooks: 
  - `KNN Implementation-1.ipynb`  
  - `KNN Implementation-2.ipynb`  
- Concepts: Distance-based classification, K-value tuning, accuracy evaluation.  

#### 🌀 Support Vector Machine (SVM)
- Notebook: `SVM Implementation-1.ipynb`  
- Concepts: Hyperplane, margin maximization, binary classification.  

#### 🌳 Decision Tree Classifier
- Notebook: `Decision Tree Classifier.ipynb`  
- Concepts: Gini index, entropy, splitting features, overfitting.  

#### 🌲 Random Forest Classifier
- Notebooks:
  - `Iris dataset classification using random forest.ipynb`  
  - `Diseases Predict using Random Forest.ipynb`  
- Concepts: Ensemble learning, bagging, feature importance.  

#### 🔬 Artificial Neural Networks (ANN)
- Notebooks:
  - `ANN Implementation-1.ipynb`  
  - `ANN Implementation-2.ipynb`  
- Concepts: Layers, activation functions, forward pass, backpropagation.  

---

### 📊 **4. Datasets Included**

| Dataset | File | Usage |
|---------|------|-------|
| Iris Dataset | `iris.csv` | Flower classification |
| Student Info | `student_info.csv` | Student marks prediction |
| Insurance Dataset | `insurance.csv` | Cost prediction |
| Sports Dataset | `sports.csv` | Classification tasks |
| Disease Dataset | `dataset.csv`, `improved_disease_dataset.csv` | Disease prediction |
| Taxi Dataset | `taxi.csv` | Regression/classification practice |
| Employee Dataset | `Emp.csv` | Categorical encoding practice |

---

### 💾 **5. Saved Models**
- `best_model.pkl` → Best performing model  
- `dtc.pkl` → Decision tree classifier  
- `label_encoder.pkl` → Encoders for categorical features  
- `smp.pkl` → Student marks predictor  

These models can be **loaded back using `pickle`** for predictions without retraining.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/ml-basics.git
cd ml-basics

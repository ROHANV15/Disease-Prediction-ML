
# 🧠 Disease Prediction System using Machine Learning (Diabetes Prediction)

This project demonstrates how machine learning can be applied to predict the likelihood of diabetes in patients based on medical features. It uses logistic regression for classification and evaluates model performance using key metrics and visualizations.

## 📌 Project Objective

To build an AI/ML-based disease prediction system using supervised learning techniques to **predict whether a patient is diabetic** based on diagnostic features. This type of system can support early diagnosis and preventive healthcare.

## 📊 Dataset Information

The dataset used for this project contains various health-related metrics collected from patients. Each row represents a patient, and each column represents a medical indicator.

**Features:**

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

**Label (Target):**

- `Outcome` → 0 (Non-diabetic), 1 (Diabetic)

> Ensure the file `dataset.csv` is available in your project directory before running the code.

## ⚙️ Technologies & Libraries Used

- **Python 3.x**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## 📁 Project Structure


isease-prediction/
├── dataset.csv
├── diabetes_prediction.py
└── README.md

## 🚀 How to Run

✅ Step 1: Install Required Packages

Open your terminal and run:
pip install pandas numpy matplotlib seaborn scikit-learn

✅ Step 2: Clone or Download the Repository

You can either download the ZIP or clone it:
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction

✅ Step 3: Run the Code

You can run the script in VS Code or in a Jupyter Notebook:

python diabetes_prediction.py

The program will:

Train a logistic regression model

Predict test results

Display a confusion matrix

Plot a ROC curve

Print evaluation metrics like accuracy, precision, recall, and F1-score

##📈 Model Performance Metrics

After training, the following metrics are calculated and displayed:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC Score
These give a complete picture of how well the model is performing.

##📉 Visualization Examples

🔷 Confusion Matrix:
Visualizes how many predictions were correct vs. incorrect for each class.

🔷 ROC Curve:
Shows the trade-off between sensitivity (recall) and specificity.

🎯 Summary


This project serves as a foundational example for medical AI systems. It demonstrates:
How to preprocess and analyze health data
How to train and evaluate a classification model
How to interpret results using both metrics and visualizations
It can be extended to include:
Hyperparameter tuning
Multiple models (e.g., Random Forest, SVM)
Web integration using Flask or Streamlit

📄 License

This project is open for educational and research purposes.

✨ Acknowledgments

Special thanks to open-source contributors and dataset providers. This work is part of an academic portfolio for the M.E. Computer Science and Engineering program.

🔗 Author
Rohan V

📧 [rohanvoff@gmail.com]


🔗 GitHub: https://github.com/ROHANV15
=======
# Disease-Prediction-ML
Machine Learning based Disease Prediction System (Diabetes/Heart/Cancer)
>>>>>>> 3f3407eb77f7bae71bfa927a8f28db9cbd0e2e3b
 

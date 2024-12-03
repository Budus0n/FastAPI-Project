Early Sepsis Prediction Using Machine Learning
This project leverages machine learning to predict sepsis in ICU patients using clinical data. By identifying sepsis early, we aim to assist healthcare professionals in making timely interventions, ultimately improving patient outcomes.
________________________________________
📋 Table of Contents
1.	Project Overview
2.	Dataset Description
3.	Setup Instructions
4.	Methodology
5.	Results
6.	Contributing
7.	License
________________________________________
🌟 Project Overview
Problem Statement:
Sepsis is a life-threatening condition caused by the body's response to infection. Early detection is critical for effective treatment.
Goals:
•	Develop a machine learning model to predict sepsis based on clinical features.
•	Ensure high model performance (precision, recall, F1-score).
Key Business Questions:
1.	What is the class distribution of sepsis in the dataset?
2.	What are the most important features influencing predictions?
3.	Can hyperparameter tuning enhance model performance?
Framework:
The CRISP-DM (Cross-Industry Standard Process for Data Mining) was adopted for this project.
________________________________________
🗂️ Dataset Description
The dataset includes clinical data for ICU patients. Each row represents a patient, with features such as plasma glucose, blood pressure, and age.
Key Columns:
•	PRG: Plasma Glucose
•	PL: Blood Work Result-1
•	PR: Blood Pressure
•	SK: Blood Work Result-2
•	Age: Patient's Age
•	Sepsis: Target variable (1 = Sepsis, 0 = No Sepsis)
Preprocessing Steps:
1.	Handling missing and duplicate values.
2.	Outlier treatment and feature scaling.
3.	Feature selection and engineering.
________________________________________
🛠️ Setup Instructions
1. Clone the Repository
bash
Copy code
git clone https://github.com/Budus0n/FastAPI-Project
cd sepsis-prediction
2. Install Dependencies
pip install -r requirements.txt
3. Run the Scripts
1.	Data Preprocessing:
python data_preprocessing.py
2.	Model Training and Evaluation:
python train_model.py
3.	Hyperparameter Tuning (Optional):
python hyperparameter_tuning.py
________________________________________
🔍 Methodology
1. Data Exploration
•	Visualized distributions of clinical features.
•	Analyzed class imbalances and correlations.
2. Data Preparation
•	Median imputation for missing values.
•	RobustScaler for outlier mitigation.
•	QuantileTransformer for feature normalization.
3. Model Building
Trained the following models:
•	Logistic Regression
•	Random Forest
•	K-Nearest Neighbors
•	LightGBM
•	Support Vector Machines (SVM)
4. Hyperparameter Tuning
Optimized key models using GridSearchCV to improve accuracy and recall.
5. Feature Importance
Used Random Forest and LightGBM to rank features contributing to sepsis predictions.
________________________________________
📈 Results
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.85	0.84	0.86	0.85
Random Forest	0.88	0.87	0.89	0.88
K-Nearest Neighbors	0.81	0.80	0.82	0.81
LightGBM	0.89	0.88	0.90	0.89
Support Vector Machine	0.87	0.86	0.88	0.87
The LightGBM model achieved the best overall performance and was saved for deployment.
________________________________________
🤝 Contributing
We welcome contributions!
To contribute:
1.	Fork the repository.
2.	Create a new branch (feature-branch-name).
3.	Commit your changes.
4.	Open a pull request.
________________________________________
🛡️ License
This project is licensed under the MIT License.
________________________________________
✨ Contact
For questions or support, please reach out to Buduson.


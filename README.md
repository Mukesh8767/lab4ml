##Lab 4: Support Vector Machine (SVM) and Support Vector Regression (SVR)
This repository contains the code and explanations for implementing Support Vector Machine (SVM) for classification. The lab demonstrates how to apply the SVM model to classify data and tune hyperparameters using GridSearchCV.

Aim
To study and implement Support Vector Machine (SVM) for classification.
To perform Support Vector Regression (SVR) (Optional based on your further implementation).
Files
SVM_05.ipynb: The Jupyter notebook containing the SVM implementation for classifying Parkinson's disease status using the parkinsons_new.csv dataset.
Dataset
The dataset used for this lab is parkinsons_new.csv. It contains features related to Parkinson's disease diagnosis.

Features: Various attributes including voice measurements.
Target: status (binary classification indicating disease presence: 0 or 1).
Prerequisites
To run the code in this repository, you need the following installed:

Python 3.x
Jupyter Notebook or Jupyter Lab
The following Python libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
You can install these packages using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn
Usage
Clone the repository to your local machine:
bash
Copy code
git clone https://github.com/yourusername/svm-lab.git
Open the Jupyter notebook:
bash
Copy code
jupyter notebook SVM_05.ipynb
Run the cells in the notebook to perform SVM classification on the Parkinson's dataset.
Steps in the Notebook
Data Loading and Preprocessing:

Load the dataset parkinsons_new.csv and perform exploratory data analysis (EDA).
Clean the dataset by removing irrelevant columns (e.g., name) and handling categorical variables using one-hot encoding.
Split the dataset into training and testing sets.
Feature Scaling:

Apply StandardScaler to normalize the features for better performance of SVM.
Training the SVM Model:

Train the SVM classifier using a linear kernel and test its performance using accuracy and confusion matrix.
Hyperparameter Tuning with GridSearchCV:

Use GridSearchCV to find the best hyperparameters (e.g., C, degree, gamma, kernel) for the SVM model.
Print the best parameters and evaluate the model using a classification report.
Model Evaluation:

Evaluate the trained model using various metrics like accuracy, confusion matrix, and classification report.
Example Output
After running the notebook, you will see:

SVM model predictions for both the training and testing datasets.
Performance metrics such as accuracy and confusion matrix.
Best hyperparameters selected by GridSearchCV and a detailed classification report.
Conclusion
In this lab, we learned how to:

Implement SVM for classification.
Preprocess the data and perform feature scaling.
Evaluate the SVM model's performance using accuracy, confusion matrix, and classification report.
Tune SVM hyperparameters using GridSearchCV for better performance.

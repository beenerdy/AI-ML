# Loan Default Prediction

This repository contains the code and resources for predicting loan defaults using historical data. The primary objective is to create a machine learning model that can predict whether a loan will default based on various borrower characteristics. This project was completed as part of an assignment and leverages techniques in data preprocessing, exploratory data analysis (EDA), and deep learning.

## Project Overview

The aim of this project is to predict whether a loan will be fully paid or will default, using a dataset spanning from 2007 to 2015. For financial institutions, predicting loan defaults is crucial for managing risk and making informed lending decisions.

### Dataset

The dataset contains various features related to the loan and the borrower, such as the loan purpose, credit score (FICO), and interest rate. The target variable is binary, indicating whether the loan was fully paid or not.

## Key Concepts and Techniques

### 1. **Exploratory Data Analysis (EDA)**
   - **Imbalanced Data**: 
     - The dataset is highly imbalanced, with approximately 83% of loans being fully paid and only 16% being defaults. This imbalance is a critical factor to consider during model training, as it can lead to biased predictions.
   - **Visualization**:
     - Various plots, such as count plots and bar plots, were used to understand the distribution of the data and the relationship between different features and the target variable.
   - **Correlation Analysis**:
     - Features were analyzed for correlations with the target variable, especially focusing on variables like FICO score and interest rate, which are key indicators of loan repayment success.

### 2. **Data Preprocessing**
   - **Handling Categorical Variables**:
     - Categorical features, such as the loan purpose, were transformed into numerical representations using techniques like one-hot encoding to make them suitable for model training.
   - **Oversampling**:
     - To address the data imbalance, the minority class (loans not fully paid) was oversampled using Pandas’ `sample` method. This ensures that the model does not become biased towards the majority class.
   - **Feature Engineering**:
     - Highly correlated features were identified and, where appropriate, removed to simplify the model and avoid multicollinearity. For instance, the interest rate was removed due to its high inverse correlation with the FICO score.

### 3. **Model Building**
   - **Deep Learning Model**:
     - A neural network was constructed with 19 neurons in the first layer, corresponding to the 19 features after preprocessing. Dropout layers with a rate of 0.2 were added to reduce overfitting by preventing the network from relying too heavily on any particular neurons.
   - **Loss Function**:
     - Binary cross-entropy was used as the loss function, suitable for binary classification tasks.
   - **Early Stopping**:
     - Early stopping was implemented as a form of regularization, halting training when no significant improvement in validation accuracy was observed to prevent overfitting.

### 4. **Model Evaluation**
   - **Accuracy**:
     - The final model achieved an accuracy of approximately 65%, indicating a moderate ability to correctly classify loan repayment outcomes.
   - **Error Analysis**:
     - Given the context of the problem, reducing Type II errors (false negatives, where a defaulting loan is predicted as fully paid) was prioritized, even at the expense of increasing Type I errors.

## How to Run the Project

### Prerequisites
- Python 3.x
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Jupyter Notebook (optional, for interactive exploration)

### Steps to Run
1. Navigate to the project directory:
   ```bash
   cd loan-default-prediction
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Assignment.ipynb"
   ```

## Results and Interpretation

The model demonstrates moderate accuracy in predicting loan defaults. Given the challenges posed by the imbalanced dataset, future work could involve further tuning of the model and exploring additional features or more complex models to improve prediction accuracy.

## Future Work
- **Hyperparameter Tuning**: Experiment with different network architectures, learning rates, and regularization techniques to improve model performance.
- **Alternative Models**: Explore more sophisticated models, such as Random Forests or Gradient Boosting Machines, which might handle imbalanced data more effectively.
- **Feature Exploration**: Investigate additional features or external data sources that might enhance the predictive power of the model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Udacity for providing the coursework and guidance.
- The open-source community for providing the tools and libraries used in this project.

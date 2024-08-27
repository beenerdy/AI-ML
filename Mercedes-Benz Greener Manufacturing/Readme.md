# Mercedes-Benz Greener Manufacturing

This repository contains the code and resources for optimizing the time cars spend on the test bench during the manufacturing process at Mercedes-Benz. The project focuses on reducing the testing time for various configurations of Mercedes-Benz cars, ultimately contributing to lower carbon emissions without compromising on safety and reliability.

## Project Overview

The primary objective of this project is to develop a predictive model that reduces the time a car spends on the test bench by analyzing various car features. The project involves data preprocessing, feature selection, and the application of machine learning techniques to build a model that can accurately predict the testing time.

### Problem Statement

Mercedes-Benz, a leader in the premium car industry, offers a wide range of customizable features and options for its vehicles. To ensure the safety and reliability of each unique car configuration, the company employs a robust testing system. However, optimizing the speed of this testing process is complex and time-consuming. By leveraging machine learning, the goal is to reduce testing time, which will help decrease carbon dioxide emissions while maintaining high standards of safety and quality.

## Key Concepts and Techniques

### 1. **Data Preprocessing**
   - **Variance Thresholding**: 
     - Features with zero variance across all samples were identified and removed, as they do not contribute to the predictive power of the model.
   - **Null and Unique Value Checks**:
     - The dataset was checked for missing values and the number of unique values in each feature to ensure data quality and consistency before model training.
   - **Label Encoding**:
     - Categorical variables were converted into numerical format using label encoding, which is essential for machine learning algorithms to process non-numeric data.

### 2. **Dimensionality Reduction**
   - **Feature Selection**:
     - Dimensionality reduction techniques were applied to select the most relevant features, reducing the complexity of the model and improving its performance. This step is critical given the high dimensionality of the dataset with 377 features.

### 3. **Model Building**
   - **XGBoost**:
     - XGBoost, an efficient and scalable implementation of gradient boosting, was used to build the predictive model. XGBoost is known for its high performance in classification and regression tasks, making it well-suited for this problem.
   - **Model Training**:
     - The model was trained on the preprocessed dataset, and various hyperparameters were tuned to optimize performance.

### 4. **Prediction and Evaluation**
   - The trained XGBoost model was used to predict the testing time on a separate test dataset. The model's performance was evaluated based on its ability to accurately predict the time required for each car configuration to pass the testing process.

## How to Run the Project

### Prerequisites
- Python 3.x
- Pandas
- Scikit-learn
- XGBoost
- Jupyter Notebook (optional, for interactive exploration)

### Steps to Run
1. Navigate to the project directory:
   ```bash
   cd mercedes-benz-greener-manufacturing
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Project.ipynb"
   ```

## Results and Interpretation

The XGBoost model effectively reduced the testing time for various car configurations. By selecting the most relevant features and applying a powerful machine learning algorithm, the model contributes to the overall goal of reducing carbon emissions in the manufacturing process.

## Future Work
- **Hyperparameter Optimization**: Further tuning of the model's hyperparameters could lead to even better performance.
- **Feature Engineering**: Additional feature engineering techniques could be applied to uncover more predictive power in the data.
- **Model Comparison**: Experiment with other machine learning models, such as Random Forests or Neural Networks, to compare performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mercedes-Benz for providing the problem statement and data.
- The open-source community for the tools and libraries used in this project.

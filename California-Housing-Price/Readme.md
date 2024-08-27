# California Housing Price Prediction

This repository contains the code and resources for predicting housing prices in California using linear regression. The goal of this project is to predict the median house values in various districts of California based on a provided dataset.

## Project Overview

The primary objective of this project is to build a predictive model using linear regression that can estimate the median housing prices in California. The dataset used for this task includes various features such as the number of bedrooms, population, and proximity to the ocean, which are crucial for predicting house prices.

## Key Concepts and Techniques

### 1. **Data Acquisition and Exploration**
   - **Dataset Import**: The dataset is imported using Pandas into a DataFrame for exploration and processing. 
   - **Initial Exploration**: Functions like `df.head()` and `df.shape()` are used to understand the structure of the dataset, including the number of rows and features.

### 2. **Data Preprocessing**
   - **Handling Missing Values**: 
     - The dataset includes some missing values in the `total_bedrooms` feature. These missing values are imputed using the mean of the respective column with the help of `SimpleImputer` from Scikit-learn.
   - **Categorical Feature Encoding**:
     - The `ocean_proximity` feature, which is categorical, is encoded into numerical values using `pd.get_dummies()`. Before encoding, the `ISLAND` category, which has very few entries, is merged with the `NEAR_BAY` category to simplify the data.
   - **Feature Scaling**:
     - Features are standardized using `StandardScaler` to ensure that they have a mean of 0 and a standard deviation of 1, which is crucial for linear regression models.

### 3. **Data Visualization**
   - **Histogram Visualization**:
     - Histograms for each feature are plotted to understand their distributions. Many features exhibit right-skewed distributions, which is typical for real estate data (e.g., fewer properties with very high values).

### 4. **Model Building**
   - **Linear Regression**:
     - Linear regression is chosen as the model due to the problem's supervised nature, where the goal is to predict a continuous target variable (`median_house_value`).
   - **Train-Test Split**:
     - The data is split into training (80%) and testing (20%) sets to evaluate the model's performance on unseen data.
   - **Model Training**:
     - The model is trained on the training set, and its performance is evaluated on the test set. The initial model achieves an R^2 score of approximately 63%, indicating a moderate fit.

### 5. **Model Evaluation**
   - **Root Mean Squared Error (RMSE)**:
     - RMSE is calculated to quantify the average magnitude of the error in predictions. It measures the standard deviation of the residuals (prediction errors), providing insight into how well the model's predictions match the actual data.
   - **Model Visualization**:
     - A scatter plot is generated to compare the actual test values (`Y_test`) against the predicted values. The plot suggests a linear relationship, supporting the appropriateness of linear regression for this task.

## Results and Interpretation

The linear regression model provides a moderately accurate prediction of housing prices in California, with an R^2 score of about 63%. The RMSE gives further insight into the prediction errors, allowing for future improvements such as using more complex models or additional features.

## Future Work
- **Feature Engineering**: Explore additional features or transformations that could improve the model's predictive power.
- **Advanced Modeling**: Consider using more sophisticated models like decision trees, random forests, or gradient boosting to improve prediction accuracy.
- **Hyperparameter Tuning**: Implement grid search or random search for optimizing the linear regression model parameters.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Scikit-learn documentation and tutorials.
- Open-source contributors for their invaluable tools and resources.

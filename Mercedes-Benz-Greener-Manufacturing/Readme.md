# Mercedes-Benz Greener Manufacturing

This project aims to reduce the time cars spend on the test bench during the manufacturing process at Mercedes-Benz. By developing a predictive model, the goal is to minimize testing time for various car configurations, ultimately lowering carbon emissions while maintaining safety and reliability standards.

## Project Overview

In this project, I worked on optimizing the time cars spend on the test bench by training a machine learning model to predict the testing time based on each car's configuration. This involved data preprocessing, feature selection, and the use of XGBoost to build an efficient model.

The ultimate goal is to contribute to Mercedes-Benz's sustainability efforts by reducing the time needed for testing, thus lowering carbon emissions while maintaining safety standards.

### Problem Statement

Mercedes-Benz offers a wide range of customizable vehicle options, requiring a robust testing process to ensure safety and reliability. However, this process is time-consuming. By applying machine learning, we can predict and reduce testing time, thereby optimizing the manufacturing process and lowering carbon emissions.

## Key Concepts and Techniques

### 1. Data Integrity and Preprocessing
- **No missing or duplicated values**: The dataset was clean, with no missing or duplicated entries.
- **ID checks**: The IDs were confirmed to be random, with no correlation to other features.
- **Target Value Range**: The testing time ranged between 72 and 265 seconds, with most values between 75 and 150 seconds.
- **Outlier Detection**: A scatter plot and percentile analysis identified an outlier with a test time of 265 seconds. Rows with test times greater than 155 seconds were removed.
- **Categorical and Binary Features**:
  - 8 categorical features were analyzed using box plots to assess variance.
  - 368 binary features were analyzed for variance, with 13 zero-variance and 53 duplicated features removed.
  
### 2. Feature Selection and Encoding
- Categorical variables were encoded using a custom `preprocess_categorical` function with LabelEncoder.
- PCA (Principal Component Analysis) was applied to reduce dimensionality from 309 features to 10, excluding categorical features.

### 3. Model Building
- **XGBoost**: Chosen for its efficiency, XGBoost was trained on the dataset, with hyperparameter optimization performed using GridSearch.
- **Performance**: The model achieved an R² score of 67%. However, further training attempts encountered resource constraints, and the full model was not retrained.

### 4. Prediction and Evaluation
- The model was evaluated using the R² score and tested on new car configurations to predict their testing time.

## Results

The XGBoost model successfully reduced the predicted testing time for various car configurations. By selecting the most relevant features and applying machine learning techniques, this project contributes to greener manufacturing by lowering emissions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

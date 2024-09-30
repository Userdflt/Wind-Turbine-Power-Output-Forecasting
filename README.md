# Wind-Turbine-Power-Output-Forecasting

![Diagram](landscape-with-windmills.jpg)

[CLICK FOR NOTEBOOK - CODE](Power_Output_Forecasting_Regression.ipynb)

# Problem Statement

The objective of this project is to forecast power output using historical environmental and temporal data over a three-year period. By employing regression techniques, we aim to predict continuous values of power based on features such as wind speed, temperature, pressure, and time-related variables. Accurate power forecasting is crucial for efficient energy management, cost reduction, and optimizing operations in the energy sector.

# Industry/Domain Context

In the energy industry, particularly within renewable energy sectors like wind and solar power, the ability to accurately predict power output is essential. Variability in environmental conditions can lead to fluctuations in power generation. Energy companies and grid operators rely on precise forecasts to balance supply and demand, enhance grid reliability, and make informed decisions regarding energy distribution and storage.

# Business Question

- **How can we accurately predict power output using historical environmental and temporal data over a three-year period?**

# Data Question

- **Which features significantly influence power output, and how can we preprocess and model the data to achieve optimal predictive performance?**

# Data Overview

- **Datasets Used:**
    - `Train.csv`: Training dataset containing historical records.
    - `Test.csv`: Testing dataset for evaluating model performance.
    - `column_info.csv`: Descriptions of the variables in the datasets.
- **Key Variables:**
    - **Temporal Features:** `Time`, `Month`, `Year`, `TimeOfDay`.
    - **Environmental Variables:** `WS_10m`, `WS_100m` (wind speeds at 10m and 100m), `Temperature`, `Pressure`, among others.
    - **Target Variable:** `Power` (the power output to be predicted).

# Process

## Data Preprocessing

1. **Time Feature Transformation:**
    - Converted the `Time` column to datetime format.
    - Extracted new features: `Month`, `Year`, and `TimeOfDay` (hour of the day).
    - Dropped the original `Time` and `Unnamed: 0` columns as they were no longer needed.
2. **Handling Missing Values and Duplicates:**
    - Checked for null values and duplicates in both training and testing datasets.
    - Found zero null values and zero duplicates, indicating good data quality.
3. **Feature Selection:**
    - Calculated the correlation matrix to identify highly correlated features.
    - Dropped the `DP_2m` column due to its high correlation with other variables, reducing multicollinearity and improving model performance.
4. **Feature Scaling:**
    - Applied `MinMaxScaler` to normalize numerical features, ensuring that all features contribute equally to the model training process.

## Exploratory Data Analysis

1. **Distribution Analysis:**
    - Plotted histograms for all features to understand their distributions.
    - Observed that most features had a normal distribution, suitable for regression modeling.
2. **Pairwise Relationships:**
    - Created pair plots to visualize relationships between features and the target variable.
    - Noted strong relationships between wind speed features and power output.
3. **Temporal Analysis:**
    - Analyzed how `Power` varies with `TimeOfDay`, `Month`, and `Year`.
    - Found that power output had noticeable patterns based on the time of day and month, indicating temporal dependencies.
4. **Wind Speed Analysis:**
    - Investigated monthly average wind speeds at 10m and 100m above the surface.
    - Identified that higher wind speeds generally correlated with higher power output.
5. **Correlation Analysis:**
    - Calculated the correlation coefficients between all features and `Power`.
    - Features like `WS_10m`, `WS_100m`, and `TimeOfDay` showed strong positive correlations with power output.

# Modeling

## Model Training & Evaluation

1. **Data Splitting:**
    - Split the preprocessed data into training and testing sets with an 80-20 split.
    - Ensured that the model is evaluated on unseen data to assess its generalization capabilities.
2. **Models Evaluated:**
    - **Linear Regression**
    - **Lasso Regression**
    - **Ridge Regression**
    - **K-Nearest Neighbors (KNN) Regressor**
    - **Decision Tree Regressor**
    - **Random Forest Regressor**
    - **AdaBoost Regressor**
3. **Model Evaluation Metrics:**
    - **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.
    - **R-squared (R²) Score:** Indicates the proportion of variance in the dependent variable predictable from the independent variables.
4. **Training and Evaluation Results:**
    
    
    | Model | Training MAE | Training R² Score | Test MAE | Test R² Score |
    | --- | --- | --- | --- | --- |
    | Linear Regression | 0.1384 | 0.505355 | 0.1403 | 0.492294 |
    | Lasso Regression | 0.2125 | 0.000000 | 0.2122 | -0.000039 |
    | Ridge Regression | 0.1384 | 0.505339 | 0.1403 | 0.492322 |
    | **K-Nearest Neighbors** | **0.0538** | **0.897932** | **0.0731** | **0.824192** |
    | Decision Tree Regressor | 0.0000 | 1.000000 | 0.1118 | 0.561457 |
    | Random Forest Regressor | 0.0318 | 0.969111 | 0.0858 | 0.777471 |
    | AdaBoost Regressor | 0.1494 | 0.498542 | 0.1500 | 0.492723 |
5. **Predictions and Residual Analysis (K-Nearest Neighbors Regressor):**
    - After identifying the KNN Regressor as the best-performing model, we analyzed its predictions in more detail.
    
    | Index | Actual Value | Predicted Value | Difference |
    | --- | --- | --- | --- |
    | 24950 | 0.387496 | 0.376016 | 0.011480 |
    | 17553 | 0.692396 | 0.322356 | 0.370040 |
    | 19265 | 0.580596 | 0.692156 | -0.111560 |
    | 32546 | 0.169296 | 0.166896 | 0.002400 |
    | 67423 | 0.001196 | 0.006836 | -0.005640 |
    | ... | ... | ... | ... |
    | 26519 | 0.641096 | 0.817176 | -0.176080 |
    | 213 | 0.909396 | 0.911676 | -0.002280 |
    | 131967 | 0.100796 | 0.054536 | 0.046260 |
    | 11856 | 0.589596 | 0.598836 | -0.009240 |
    | 94438 | 0.225896 | 0.292056 | -0.066160 |
    - **Mean Difference:** The average of the differences between the actual and predicted values is **0.0012**, indicating that the model's predictions are, on average, very close to the actual values.
6. **Residual Analysis:**
    - **Mean Difference Interpretation:**
        - A mean difference of **0.0012** suggests that the KNN model is highly accurate, with minimal bias in its predictions.
        - The small average error supports the high R² Score and low MAE obtained during evaluation.
    - **Variance in Differences:**
        - While the mean difference is low, some individual predictions have larger residuals (e.g., differences of 0.370040 and -0.176080).
        - This indicates that while the model performs well overall, there may be specific instances where the prediction error is higher, possibly due to outliers or atypical observations.

# Comparisons and Improvements

1. **Model Performance Comparison:**
    - **K-Nearest Neighbors Regressor** outperformed other models, achieving:
        - **Training MAE:** 0.0538
        - **Training R² Score:** 0.897932
        - **Test MAE:** 0.0731
        - **Test R² Score:** 0.824192
    - **Linear Regression, Ridge, and AdaBoost Regressor** showed moderate performance but were not as effective as KNN.
    - **Lasso Regression** underperformed, possibly due to the penalty term eliminating important features.
    - **Decision Tree Regressor** exhibited overfitting, with perfect training performance but significantly lower test performance.
    - **Random Forest Regressor** showed good performance but was slightly less accurate than KNN.
2. **Insights:**
    - The superior performance of the KNN Regressor suggests that local patterns and nearest neighbors in the feature space are significant for predicting power output.
    - The minimal mean difference between actual and predicted values highlights the model's accuracy and reliability.
3. **Model Fine-Tuning:**
    - **KNN Regressor:**
        - Experimenting with different values of `n_neighbors` and weighting schemes could further enhance performance.
        - Cross-validation can be used to find the optimal parameters.
    - **Random Forest Regressor:**
        - Tuning hyperparameters such as the number of trees, tree depth, and minimum samples per leaf might improve accuracy and reduce overfitting.

# Outcomes

1. **Feature Importance:**
    - Wind speeds (`WS_10m`, `WS_100m`) and temporal features (`TimeOfDay`, `Month`) significantly influence power output.
    - These features enable the model to capture both environmental conditions and temporal trends affecting power generation.
2. **Model Efficacy:**
    - The **K-Nearest Neighbors Regressor** achieved:
        - **Test R² Score:** 82.42%
        - **Test MAE:** 0.0731
        - **Mean Difference:** 0.0012
    - These metrics indicate that the model explains a significant portion of the variance in power output and predicts with high accuracy.
3. **Business Implications:**
    - Accurate power forecasting allows for:
        - Optimized energy production schedules.
        - Improved grid reliability.
        - Informed decision-making for resource allocation.
    - This leads to cost savings and increased operational efficiency in the energy sector.

# Data Answer

By preprocessing the data to extract relevant temporal features and normalizing numerical variables, we enhanced the model's ability to learn from the data. The **K-Nearest Neighbors Regressor** effectively utilized these features to predict power output with high accuracy, as evidenced by the low mean difference of **0.0012** between actual and predicted values. Key influencing factors were identified, and the model demonstrates strong generalization to unseen data.

# Business Answer

Implementing the **K-Nearest Neighbors regression model** allows for accurate forecasting of power output based on environmental and temporal data. The model's high accuracy and minimal prediction errors enable the business to optimize energy production schedules, improve grid reliability, and make informed decisions. This ultimately leads to cost savings and increased efficiency in operations, strengthening the organization's competitive advantage in the energy sector.

# End-to-End Solution

The project delivers a comprehensive solution encompassing:

- **Data Ingestion and Preprocessing:** Cleaned and prepared the data for modeling by handling missing values, duplicates, and irrelevant features.
- **Feature Engineering:** Extracted and selected features that significantly impact power output, such as wind speeds and temporal variables.
- **Exploratory Data Analysis:** Gained insights into data distributions and relationships between variables, informing feature selection and model choice.
- **Model Training and Evaluation:** Trained multiple regression models and selected the best-performing model based on evaluation metrics, including MAE and R² Score.
- **Residual Analysis:** Analyzed the differences between actual and predicted values to assess the model's accuracy and identify areas for improvement.
- **Model Deployment Readiness:** The **KNN model** is ready for deployment to predict power output in real-time, aiding operational decision-making.
- **Scalability:** The approach can be scaled and adapted for larger datasets or integrated with streaming data for continuous forecasting.

---

# Conclusion

In this project, we successfully developed a predictive model to forecast power output using historical environmental and temporal data over a three-year period. The primary objective was to enhance energy management efficiency by accurately predicting continuous power output values based on factors such as wind speed, temperature, pressure, and time-related variables.

**Key Steps and Findings:**

1. **Data Understanding and Preparation:**
    - **Data Quality Assurance:** Ensured the datasets were clean, with no missing values or duplicates, which is crucial for reliable modeling.
    - **Feature Engineering:** Transformed the `Time` column into datetime format and extracted new features like `Month`, `Year`, and `TimeOfDay` to capture temporal patterns affecting power output.
    - **Multicollinearity Reduction:** Identified and removed the `DP_2m` feature due to its high correlation with other variables, enhancing model performance by reducing multicollinearity.
    - **Data Scaling:** Applied Min-Max scaling to normalize numerical features, ensuring all features contributed equally during model training.
2. **Exploratory Data Analysis (EDA):**
    - **Distribution Analysis:** Visualized the distribution of features to confirm they were suitable for regression modeling.
    - **Correlation Analysis:** Determined that wind speeds (`WS_10m`, `WS_100m`) and temporal variables (`TimeOfDay`, `Month`) had strong positive correlations with power output.
    - **Temporal Patterns:** Observed that power output exhibited noticeable patterns based on the time of day and month, indicating significant temporal dependencies.
3. **Model Training and Evaluation:**
    - **Model Selection:** Evaluated several regression models, including Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, AdaBoost, and K-Nearest Neighbors (KNN) Regressors.
    - **Best Model Identification:** The K-Nearest Neighbors Regressor outperformed other models, achieving a high R² score of approximately 82.42% on the test set and a low Mean Absolute Error (MAE) of 0.0731.
    - **Residual Analysis:** The mean difference between actual and predicted values was minimal (approximately 0.0012), indicating high accuracy and minimal bias in the KNN model's predictions.
4. **Convolution Analysis:**
    - **Pattern Recognition:** Performed convolution between each feature and the target variable `Power` to identify underlying patterns or lags not immediately apparent in the data.
    - **Visualization:** Plotted the convolution results to visualize the relationships, which could inform future feature engineering or model refinement efforts.

**Implications and Business Impact:**

- **Operational Efficiency:** The accurate forecasting model enables better scheduling of energy production, leading to optimized operations and reduced operational costs.
- **Grid Reliability:** Improved power output predictions assist in balancing supply and demand more effectively, enhancing grid stability and reliability.
- **Strategic Decision-Making:** Insights into how environmental and temporal factors affect power output empower stakeholders to make informed decisions regarding resource allocation and infrastructure investments.
- **Scalability:** The methodology and models developed can be adapted for larger datasets or real-time data streams, providing a scalable solution for ongoing power output forecasting needs.

**Future Recommendations:**

- **Model Fine-Tuning:** Further optimization of the KNN model parameters, such as experimenting with different values of `n_neighbors` and weighting schemes, could potentially enhance predictive performance.
- **Feature Expansion:** Incorporating additional relevant features, such as humidity or solar radiation, might improve the model's accuracy.
- **Real-Time Deployment:** Integrating the model into a real-time forecasting system could provide continuous insights and allow for dynamic adjustments in operations.

**Final Remarks:**

The project successfully achieved its goal of developing an accurate predictive model for power output using historical environmental and temporal data. The K-Nearest Neighbors Regressor proved to be the most effective, capturing the complex relationships within the data. This solution not only meets the immediate forecasting needs but also provides a robust foundation for future enhancements and scalability, ultimately contributing to more efficient and reliable energy management in the sector.

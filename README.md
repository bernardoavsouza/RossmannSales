# Rossmann Sales Prediction

This repository contains code and documentation for a data science project. The main goal of this project is to predict the next six weeks of Rossmann's sales. All the data used was provided by Kaggle.

# 1. Business Problem

Predicting sales requires a deep understanding of market trends, consumer behavior, and the factors that influence sales. It also requires accurate data, sophisticated analytical tools, and a willingness to adapt and adjust predictions based on new information. Overall, while traditional methods of sales forecasting may still be useful in certain situations, machine learning algorithms can provide more accurate and flexible predictions for larger and more complex datasets. Machine learning can also help automate the process of identifying key variables and adapting to changing trends, reducing the risk of human error and saving time and resources.

# 2. Business Assumptions

The main assumption considered in this project was 250,000 meters is a distance long enough to consider that there is no competitions around.

# 3. Solution Strategy

**Step 1: Data Description**

The main objective of this section was to organize and understand raw data using descriptive statistics.

**Step 2: Feature Engineering**

In this step, the hypotheses were defined and some features were created.

**Step 3: Variable Filtering**

This step was responsible for dropping customers' information since we wouldn't have it for the next six weeks. Moreover, only samples of opened shops with sales above zero were selected.

**Step 4: Exploratory Data Analysis**

This step was responsible for dropping customers' information since we wouldn't have it for the next six weeks. Moreover, only samples of opened shops with sales above zero were selected. Thereunto, three different analyses were done: univariate, bivariate, and multivariate analyses.

**Step 5: Data Preparation**

In this section, all the data was encoded and scaled to perform properly in a machine learning algorithm.

**Step 6: Feature Selection**

In this section, the Boruta algorithm was used to select the best features to be used. The selected features were chosen based on Boruta and insights gotten previously.

**Step 7: Machine Learning Modelling**

Here an average value model was considered as a benchmark since it's the simplest solution for the problem. Moreover, four models were trained and evaluated: KNN, Linear Ridge Regression, Random Forest, and XGBoost.

**Step 8: Hyperparameter Fine Tunning**

Since XGBoost seemed to be the most promising, a Random Search fine tuning algorithm was applied.

**Step 9: Results Interpretation**

This section shows a time series of the prediction, the prediction for the whole Rossmann group, and some info about errors.

**Step 10: Deployment**

In this project, a local API was coded to be used to predict remotely.

# 4. Top 3 Data Insights

**Hypotesis 3: New competitors around should decrease sales.**
**False:** New competitors raise sales.

**Hypotesis 2: Stores should sell more when they are far from competitors.**
**False:** There is no such a correlation between those variables.

**Hypotesis 1: Stores should sell more when they are in promotion.**
**False:** There is no statistical evidence that promotions raise sales

# 5. Machine Learning Model Applied

In this project, four machine learning models were tested: KNN, Linear Ridge Regression, Random Forest, and XGBoost. The final model was built using XGBoost.

# 6. Machine Learning Model Performance

The final model was built with a MAE of 764.78, MAPE of 11.51%, RMSE of 1098.38, and a MPE of -1.81%.

# 7. Business Results

The algorithm was capable of predicting sales generally six weeks forward with an average absolute error of 11.51%. Altogether, the proposed algorithm predicted a revenue of $286,501,248.00 +/- $856,000.00. Although, there are still some stores whose sales prediction is unachievable.

# 8. Conclusions

Therefore, a machine learning algorithm can be used to predict sales of Rossmann Stores with a high degree of accuracy, enabling businesses to make informed decisions about inventory management, production planning, marketing campaigns, and other key aspects of their operations.

# 9. Next Steps to Improve

The next improvement to be sought will be to develop a neural network that will do the predictions.

# Machine Learning Assignment2

## Problem statement:
How can the financial institution have a greater effectiveness for future marketing campaigns? In order to answer this, we have to analyze the last marketing campaign the bank performed and identify the patterns that will help us find conclusions in order to develop future strategies.

## Dataset Description:
The objective of this dataset is to predict whether a bank customer has opted for a term deposit based on various demographic and campaign-related features present in the bank.csv dataset. The features include age, job, marital status, balance, contact details, and previous campaign information. Using these attributes, machine learning models are trained to classify customers as either subscribing (yes) or not subscribing (no) to the deposit scheme. This helps the bank identify potential customers and improve targeted marketing strategies.

## Model Performance Comparison

| ML Model Name        | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
|----------------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression  | 0.828377 | 0.906653 | 0.831108  | 0.800303 | 0.815414 | 0.655568 |
| Decision Tree        | 0.796489 | 0.795468 | 0.790447  | 0.776097 | 0.783206 | 0.591562 |
| KNN                  | 0.778932 | 0.841185 | 0.800512  | 0.710287 | 0.752705 | 0.557361 |
| Naive Bayes          | 0.715872 | 0.807618 | 0.800910  | 0.532526 | 0.639709 | 0.444382 |
| Random Forest        | 0.857757 | 0.921905 | 0.825934  | 0.886536 | 0.855162 | 0.717499 |
| XGBoost              | 0.855249 | 0.921720 | 0.831647  | 0.870651 | 0.850702 | 0.711075 |

## Model Observation about model performance

| ML Model Name              | Observation about model performance |
|----------------------------|--------------------------------------|
| Logistic Regression        | Provided strong baseline performance with high AUC (0.9066) and balanced precision-recall values, indicating good generalization capability. |
| Decision Tree              | Moderate performance with lower AUC and MCC compared to ensemble models, suggesting possible overfitting and higher variance. |
| kNN                        | Achieved reasonable precision but lower recall, indicating it missed some positive cases; performance depends heavily on feature scaling. |
| Naive Bayes                | Lowest overall performance with significantly low recall (0.5325) and MCC, likely due to the strong independence assumption of features. |
| Random Forest (Ensemble)   | Best overall performance with highest accuracy (0.8578), AUC (0.9219), and MCC (0.7175), showing strong generalization and robustness. |
| XGBoost (Ensemble)         | Performance very close to Random Forest with high recall and AUC, demonstrating strong boosting capability and efficient handling of complex patterns. |

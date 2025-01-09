# Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification tasks. Despite its name, it is a classification algorithm, not a regression algorithm. It predicts the probability of an outcome belonging to a specific category, typically binary categories like 0 or 1, yes or no, spam or not spam.

## Logistic Regression Equations
Logistic Regression is a statistical model used for binary classification. It predicts the probability of a binary outcome (e.g., 0 or 1) based on one or more input features.

# Terminologies in Logistic Regression

| **Term**                     | **Definition**                                                                                     |
|-------------------------------|---------------------------------------------------------------------------------------------------|
| **Logistic Regression**       | A statistical method used for binary or multiclass classification problems.                      |
| **Sigmoid Function**          | A mathematical function that maps input values to probabilities in the range [0, 1].             |
| **Odds**                      | The ratio of the probability of success to the probability of failure.                           |
| **Logit Function**            | The natural log of the odds, used to model the relationship between predictors and the response.  |
| **Binary Classification**     | Classification into one of two categories, e.g., yes/no or 0/1.                                  |
| **Multinomial Classification**| Logistic regression extended for multiple classes using softmax or one-vs-rest strategies.       |
| **Coefficients (β)**          | Parameters in the logistic regression model representing the relationship between predictors and the outcome. |
| **Intercept (β₀)**            | The bias term in the logistic regression equation.                                               |
| **Maximum Likelihood Estimation (MLE)** | A method for estimating model parameters by maximizing the likelihood of observed data.       |
| **Gradient Descent**          | An optimization algorithm used to minimize the loss function by iteratively updating parameters. |
| **Cross-Entropy Loss**        | A loss function used to measure the performance of the logistic regression model.                |
| **Confusion Matrix**          | A table summarizing the performance of a classification model (TP, TN, FP, FN).                 |
| **Accuracy**                  | The ratio of correctly predicted instances to the total instances.                               |
| **Precision**                 | The ratio of true positives to the total predicted positives.                                    |
| **Recall (Sensitivity)**      | The ratio of true positives to the total actual positives.                                       |
| **F1-Score**                  | The harmonic mean of precision and recall, used to balance their trade-offs.                    |
| **ROC Curve**                 | A graph showing the true positive rate (TPR) vs. false positive rate (FPR) at various thresholds.|
| **AUC (Area Under Curve)**    | A single number summarizing the performance of the model across all classification thresholds.   |
| **Regularization**            | Techniques like L1 (Lasso) and L2 (Ridge) used to prevent overfitting by penalizing large coefficients. |
| **Overfitting**               | A scenario where the model performs well on training data but poorly on unseen data.             |
| **Underfitting**              | A scenario where the model is too simple and fails to capture the underlying patterns in the data.|
| **Feature Scaling**           | The process of normalizing or standardizing input features to improve model performance.         |
| **Multicollinearity**         | High correlation between independent variables that can affect the stability of coefficient estimates. |
| **One-vs-All (OvA)**          | A strategy for extending logistic regression to multiclass problems by training one model per class. |
| **Softmax Function**          | A generalization of the sigmoid function for multiclass classification.                         |

---



## Logistic Function – Sigmoid Function
The sigmoid function is a mathematical function used to map the predicted values to probabilities.
It maps any real value into another value within a range of 0 and 1. The value of the logistic regression must be between 0 and 1, which cannot go beyond this limit, so it forms a curve like the “S” form.
The S-form curve is called the Sigmoid function or the logistic function.
In logistic regression, we use the concept of the threshold value, which defines the probability of either 0 or 1. Such as values above the threshold value tend to 1, and those below the threshold value tend to 0.

![image](https://github.com/user-attachments/assets/205dd51d-6915-46df-835d-ffa1b2889c50)

## Terminologies involved in Logistic Regression
Here are some common terms involved in logistic regression:

### Independent variables: The input characteristics or predictor factors applied to the dependent variable’s predictions.
### Dependent variable: The target variable in a logistic regression model, which we are trying to predict.
### Logistic function: The formula used to represent how the independent and dependent variables relate to one another. The logistic function transforms the input variables into a probability value between 0 and 1, which represents the likelihood of the dependent variable being 1 or 0.
### Odds: It is the ratio of something occurring to something not occurring. it is different from probability as the probability is the ratio of something occurring to everything that could possibly occur.
### Log-odds: The log-odds, also known as the logit function, is the natural logarithm of the odds. In logistic regression, the log odds of the dependent variable are modeled as a linear combination of the independent variables and the intercept.
### Coefficient: The logistic regression model’s estimated parameters, show how the independent and dependent variables relate to one another.
### Intercept: A constant term in the logistic regression model, which represents the log odds when all independent variables are equal to zero.
### Maximum likelihood estimation: The method used to estimate the coefficients of the logistic regression model, which maximizes the likelihood of observing the data given the model.

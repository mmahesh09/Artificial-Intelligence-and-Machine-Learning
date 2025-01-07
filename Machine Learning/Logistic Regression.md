# Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification tasks. Despite its name, it is a classification algorithm, not a regression algorithm. It predicts the probability of an outcome belonging to a specific category, typically binary categories like 0 or 1, yes or no, spam or not spam.

## Logistic Regression Equations
Logistic Regression is a statistical model used for binary classification. It predicts the probability of a binary outcome (e.g., 0 or 1) based on one or more input features.

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

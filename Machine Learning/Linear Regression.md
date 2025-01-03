### What is Linear Regression?
![Link Code](https://github.com/mmahesh09/Artificial-Intelligence-and-Machine-Learning/blob/e57d1bb2c3094c1c98ba7c9ab2af84b100462189/Machine%20Learning/Code%20Snipets/linear_regression.ipynb)


Linear regression is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (predictors). It is one of the simplest and most widely used techniques in data analysis and machine learning.

---

### Types of Linear Regression

1. **Simple Linear Regression**: 
   - Involves one independent variable and one dependent variable.
   - Equation: \( y = b_0 + b_1x \), where:
     - \( y \) is the dependent variable,
     - \( x \) is the independent variable,
     - \( b_0 \) is the intercept (value of \( y \) when \( x = 0 \)),
     - \( b_1 \) is the slope (rate of change in \( y \) for a unit change in \( x \)).

2. **Multiple Linear Regression**:
   - Involves more than one independent variable.
   - Equation: \( y = b_0 + b_1x_1 + b_2x_2 + \ldots + b_nx_n \).

---

### Assumptions of Linear Regression
For linear regression to provide reliable results, the following assumptions should hold:

1. **Linearity**: The relationship between the independent and dependent variables is linear.
2. **Independence**: Observations are independent of each other.
3. **Homoscedasticity**: The variance of residuals (errors) is constant across all levels of the independent variable.
4. **Normality**: Residuals are normally distributed.
5. **No Multicollinearity**: For multiple regression, independent variables should not be highly correlated.

---

### How Linear Regression Works

1. **Model Training**:
   - Linear regression finds the best-fitting line by minimizing the sum of squared residuals (the vertical distance between the observed and predicted values).
   - The method used is called **Ordinary Least Squares (OLS)**.

2. **Prediction**:
   - Once the coefficients (\( b_0, b_1, \ldots, b_n \)) are determined, the model can predict values for new inputs using the regression equation.

---

## Example Use Cases

- **Business**: Predicting sales based on advertising spend.
- **Finance**: Forecasting stock prices or economic trends.
- **Healthcare**: Estimating patient outcomes based on medical test results.
- **Social Science**: Understanding the impact of education on income.

---

## Advantages

- Easy to implement and interpret.
- Computationally efficient.
- Provides insights into the relationships between variables.

---

## Limitations

- Sensitive to outliers, which can skew the results.
- Cannot model complex, non-linear relationships unless transformed.
- Assumptions must be met for the model to be valid.

---

![image](https://github.com/user-attachments/assets/053ac844-0876-4a7a-8162-d1ba2c9001fd)

Here Y is called a dependent or target variable and X is called an independent variable also known as the predictor of Y. There are many types of functions or modules that can be used for regression. A linear function is the simplest type of function. Here, X may be a single feature or multiple features representing the problem.

Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x)). Hence, the name is Linear Regression. In the figure above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best-fit line for our model. 

We utilize the cost function to compute the best values in order to get the best fit line since different values for weights or the coefficient of lines result in different regression lines.




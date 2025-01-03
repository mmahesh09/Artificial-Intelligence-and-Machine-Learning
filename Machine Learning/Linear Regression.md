# What is Linear Regression?

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

### Example Use Cases

- **Business**: Predicting sales based on advertising spend.
- **Finance**: Forecasting stock prices or economic trends.
- **Healthcare**: Estimating patient outcomes based on medical test results.
- **Social Science**: Understanding the impact of education on income.

---

### Advantages

- Easy to implement and interpret.
- Computationally efficient.
- Provides insights into the relationships between variables.

---

### Limitations

- Sensitive to outliers, which can skew the results.
- Cannot model complex, non-linear relationships unless transformed.
- Assumptions must be met for the model to be valid.

---


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create the dataset
data = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
    "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Separate the data into features (X) and target (y)
X = df[['Hours']]  # Hours studied
y = df['Scores']   # Scores obtained

# Split the data into training and testing sets
# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Get the model's intercept and coefficient
intercept = model.intercept_
coefficient = model.coef_[0]

# Print the regression equation
print(f"Regression Equation: Score = {intercept:.2f} + ({coefficient:.2f} * Hours)")

# Predict the score for a student studying 9.25 hours/day
hours = 9.25
predicted_score = intercept + (coefficient * hours)
print(f"Predicted score for a student studying {hours} hours/day: {predicted_score:.2f}")

# Plot the data points
plt.scatter(X, y, color='blue', label='Actual data')

# Plot the regression line
plt.plot(X, model.predict(X), color='red', label='Regression line')

# Add titles and labels
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.legend()

# Show the plot
plt.show()

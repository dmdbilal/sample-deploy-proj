# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 5  # Number of bedrooms (between 0 and 5)
y = 2 * X + 1 + np.random.randn(100, 1)  # Price of the house (with some noise)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price of the House')
plt.title('Simple Linear Regression Model')
plt.show()

# Print the model's coefficients
print(f'Coefficient (slope): {model.coef_[0][0]}')
print(f'Intercept: {model.intercept_[0]}')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = '/mnt/data/MoviesOnStreamingPlatforms.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns, if any
data.drop(columns=['Unnamed: 0'], inplace=True)

# For this example, we'll use 'Year' as a feature and 'Rotten Tomatoes' as the target for polynomial regression
# Handle missing values (if any) by filling with mean or mode for simplicity
data['Rotten Tomatoes'].fillna(data['Rotten Tomatoes'].mode()[0], inplace=True)

# Define features (X) and target variable (y)
X = data[['Year']]  # Independent variable (feature)
y = data['Rotten Tomatoes']  # Dependent variable (target)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial transformation (degree 3, you can adjust the degree as needed)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create and train the Linear Regression model on polynomial features
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

# Predict on the test data
y_pred = lin_reg.predict(X_test_poly)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Optionally, visualize the polynomial regression results
import matplotlib.pyplot as plt

plt.scatter(X, y, color='red')  # Original data points
plt.plot(X_test, y_pred, color='blue')  # Polynomial regression line
plt.title('Polynomial Regression')
plt.xlabel('Year')
plt.ylabel('Rotten Tomatoes')
plt.show()

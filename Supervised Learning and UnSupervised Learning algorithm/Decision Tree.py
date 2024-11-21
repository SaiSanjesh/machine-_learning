import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'MoviesOnStreamingPlatforms.csv'
data = pd.read_csv(file_path)

# Drop the unnecessary 'Unnamed: 0' column
data.drop(columns=['Unnamed: 0'], inplace=True)

# Handle missing values (fill missing 'Age' with 'Not Rated', 'Rotten Tomatoes' with 'No Rating')
data['Age'].fillna('Not Rated', inplace=True)
data['Rotten Tomatoes'].fillna('No Rating', inplace=True)

# Convert categorical variables (e.g., 'Age', 'Rotten Tomatoes') to numerical using Label Encoding
label_encoder = LabelEncoder()
data['Age'] = label_encoder.fit_transform(data['Age'])
data['Rotten Tomatoes'] = label_encoder.fit_transform(data['Rotten Tomatoes'])

# Define features (X) and target variable (y)
X = data[['Year', 'Age', 'Rotten Tomatoes']]
y = data['Netflix']  # Target: whether the movie is available on Netflix

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Optional: Visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Available', 'Available'], rounded=True)
plt.show()

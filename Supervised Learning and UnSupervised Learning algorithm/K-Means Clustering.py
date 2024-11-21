import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/MoviesOnStreamingPlatforms.csv'
data = pd.read_csv(file_path)

# Drop the unnecessary 'Unnamed: 0' column
data.drop(columns=['Unnamed: 0'], inplace=True)

# Handle missing values (fill missing 'Age' with 'Not Rated', 'Rotten Tomatoes' with 'No Rating')
data['Age'].fillna('Not Rated', inplace=True)
data['Rotten Tomatoes'].fillna('No Rating', inplace=True)

# Convert categorical variables (e.g., 'Age', 'Rotten Tomatoes') to numerical using Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Age'] = label_encoder.fit_transform(data['Age'])
data['Rotten Tomatoes'] = label_encoder.fit_transform(data['Rotten Tomatoes'])

# Select features for clustering (remove 'Netflix' column as it is the target for classification)
X = data[['Year', 'Age', 'Rotten Tomatoes']]

# Normalize the features using StandardScaler (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means for clustering (set k=3 as an example, adjust as needed)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the results
print(data[['Year', 'Age', 'Rotten Tomatoes', 'Cluster']].head())

# Visualize the clusters (example using Year and Age)
plt.scatter(data['Year'], data['Age'], c=data['Cluster'], cmap='viridis')
plt.title('K-Means Clustering (Year vs Age)')
plt.xlabel('Year')
plt.ylabel('Age')
plt.colorbar(label='Cluster')
plt.show()

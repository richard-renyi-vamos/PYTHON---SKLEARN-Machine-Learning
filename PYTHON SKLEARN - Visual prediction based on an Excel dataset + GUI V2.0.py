import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Step 3: Load Data
data = pd.read_excel('your_data.xlsx')

# Step 4: Preprocess Data
# For simplicity, assume all columns except the last one are features and the last one is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Machine Learning Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 6: Make Predictions
predictions = model.predict(X_test)

# Step 7: Visualize Predictions
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Small aviation dataset
data = {
    'Hour': [8, 10, 17, 19, 7, 15, 20, 12, 18, 9,
             6, 14, 21, 11, 16, 22, 13, 5, 18, 20],
    'Weather': [0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
                0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    'Congestion': [0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
                   0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    'Delayed': [0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                0, 1, 1, 0, 1, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

# Inputs (X) and output/label (y)
X = df[['Hour', 'Weather', 'Congestion']]
y = df['Delayed']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train the ML model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Predict a new flight
new_flight = pd.DataFrame({
    'Hour': [18],
    'Weather': [1],
    'Congestion': [1]
})

result = model.predict(new_flight)

if result[0] == 1:
    print("Prediction: FLIGHT DELAYED")
else:
    print("Prediction: FLIGHT ON TIME")


# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

# plt.figure(figsize=(12, 6))
# plot_tree(
#     model,
#     feature_names=X.columns,
#     class_names=["On Time", "Delayed"],
#     filled=True,
#     rounded=True
# )

# plt.show()

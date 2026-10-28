import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

# Create a sample flight dataset
data = {
    'Hour': [
        8, 10, 17, 19, 7, 15, 20, 12, 18, 9,
        6, 14, 21, 11, 16, 22, 13, 5, 18, 20
    ],
    'Weather': [
        0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 0, 1, 1
    ],
    'Congestion': [
        0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 0, 0, 1
    ],
    'Delayed': [
        0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 0, 0, 1
    ]
}

df = pd.DataFrame(data)

# Separate inputs and target
X = df[['Hour', 'Weather', 'Congestion']]
y = df['Delayed']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

# Create the two models
decision_tree = DecisionTreeClassifier(
    max_depth=3, random_state=RANDOM_STATE
)

random_forest = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=RANDOM_STATE
)

# Train both models
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Make predictions
dt_predictions = decision_tree.predict(X_test)
rf_predictions = random_forest.predict(X_test)

# Evaluate both models
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Compare results and save to a file
results = []
results.append("FLIGHT DELAY PREDICTION - MODEL COMPARISON")
results.append("=" * 45)
results.append(f"University ID / RANDOM_STATE: {RANDOM_STATE}")
results.append("")

results.append(
    f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%"
)
results.append(
    f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%"
)

results.append("\nMODEL COMPARISON")

if dt_accuracy > rf_accuracy:
    results.append(
        "Decision Tree performed better on this test set."
    )
elif rf_accuracy > dt_accuracy:
    results.append(
        "Random Forest performed better on this test set."
    )
else:
    results.append(
        "Both models achieved the same accuracy."
    )

results.append(
    "\nNote: Results are based on a small, fictional dataset."
)

# Display results
for line in results:
    print(line)

# Save results to a text file
with open("comparison.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(results))

print("\nResults saved to comparison.txt")

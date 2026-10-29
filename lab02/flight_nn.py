import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create the dataset
# Features: Hour, Weather, Congestion
# Target: 0 = On time, 1 = Delayed
X = np.array([
    [6, 0, 0],
    [7, 0, 0],
    [8, 0, 1],
    [9, 1, 1],
    [10, 0, 0],
    [11, 1, 0],
    [12, 0, 1],
    [13, 1, 1],
    [14, 0, 0],
    [15, 1, 1],
    [16, 0, 1],
    [17, 1, 1],
    [18, 0, 0],
    [19, 1, 0],
    [20, 0, 1],
    [21, 1, 1],
    [7, 1, 0],
    [10, 1, 1],
    [14, 1, 0],
    [18, 1, 1]
], dtype=np.float32)
y = np.array([
    0, 0, 0, 1, 0,
    1, 0, 1, 0, 1,
    0, 1, 0, 0, 1,
    1, 0, 1, 0, 1
], dtype=np.float32)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Build the neural network
model = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.summary()

# Configure and train the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    X_train,
    y_train,
    epochs=50,
    verbose=0
)
print("\nTraining completed.")

# Evaluate the model
y_probability = model.predict(X_test, verbose=0).ravel()
y_pred = (y_probability >= 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nActual outcomes:    ", y_test.astype(int))
print("Predicted outcomes: ", y_pred)
print("Test accuracy:      ", round(accuracy, 3))
print("\nConfusion matrix:")
print(matrix)

# Predict the outcome of a new fictional flight
# Features: Hour, Weather, Congestion
new_flight = np.array([[17, 1, 1]], dtype=np.float32)
probability = model.predict(new_flight, verbose=0)[0, 0]
if probability >= 0.5:
    prediction = "DELAYED"
else:
    prediction = "ON TIME"

print("\nNew flight:", new_flight[0])
print("Estimated probability of delay:", round(float(probability), 3))
print("Predicted outcome:", prediction)

# Educational demonstration only.
# This model is not suitable for real aviation operations.

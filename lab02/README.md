# Deep Learning Fundamentals for Aviation

## Lab Activity: Flight Outcome Classification Using a Neural Network

### Objective

In this laboratory activity, you will build and train a simple artificial neural network to classify flight outcomes using Python and TensorFlow/Keras.

By completing this activity, you will learn how to:

* Build a neural network with input, hidden, and output layers.
* Understand the role of activation functions.
* Train a neural network using sample data.
* Evaluate predictions using accuracy and a confusion matrix.
* Use a trained model to predict the outcome of a new flight.
* Identify the limitations of a model trained on a small synthetic dataset.

### Scenario

An airline wants to explore whether basic operational information can help classify flights as delayed or on time.

For this exercise, you will use three input features:

| Feature |	Description | Example |
| - | - | - |
| Hour | Scheduled departure hour | 17 |
| Weather | Simplified weather indicator | 0 = Favorable, 1 = Unfavorable |
| Congestion | Simplified airport congestion indicator | 0 = Low, 1 = High |

The target variable is:

| Value | Flight outcome |
| - | - |
| 0 | On time |
| 1 | Delayed |

The dataset is fictional and is intended exclusively for educational purposes.

### Required Software

You will need:

* Python 3
* Google Colab or Jupyter Notebook
* NumPy
* TensorFlow/Keras
* Scikit-learn

Install the required libraries if they are not already available:

```
pip install numpy tensorflow scikit-learn
```

### Code Provided

You will receive the following file, which contains the code:

* `flight_nn.py` — the Python script used to build and train the flight neural network.

Read the instructions before running the Python script.

### Activity 1 — Understand the Dataset

Open `flight_nn.py` and locate the section where the dataset is defined.

Examine the input features and target values.

Answer the following questions:
1. What are the three input features?
2. How many neurons are in the hidden layer?
3. What is the purpose of the ReLU and sigmoid activation functions?

### Activity 2 — Train the Model

Run the Python script and observe the training and evaluation process.

Complete the following tasks:
1. Identify the optimizer and loss function used by the model.
2. Record the test accuracy.
3. Change the number of training epochs from 50 to 10, run the script again, and compare the results.

### Activity 3 — Predict a Flight Outcome

Use the script to predict the outcome of a new fictional flight.

Complete the following tasks:

1. Record the predicted probability and classification.
2. Change the input values to represent a different flight.
3. Run the prediction again and compare the results.

### Activity 4 — Evaluate the Results

Examine the model’s accuracy and confusion matrix.

Answer the following questions:

1. How many test flights were classified correctly?
2. Did the model incorrectly classify any delayed flights as on time?
3. Why is accuracy alone insufficient to evaluate an aviation-related prediction model?
4. Suggest two additional features that could help a more realistic flight-delay prediction model.

### Submission Requirements

Submit:
* A short report containing your answers, test accuracy, confusion matrix, and one example prediction.
* A brief conclusion explaining the limitations of the model.

**Important!** The dataset is small and synthetic. This exercise demonstrates the neural network workflow and does not produce a model suitable for real aviation operations.

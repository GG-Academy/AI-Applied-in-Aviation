# Applied Machine Learning for Aviation

## Comparing Decision Tree and Random Forest Models

This course activity demonstrates how two *Machine Learning* algorithms can be used to predict flight delays using a small, fictional dataset. Students will train and evaluate a Decision Tree Classifier and a Random Forest Classifier using the same flight data, then compare their prediction accuracy. The exercise introduces model comparison and highlights the importance of evaluating Machine Learning models on unseen data.

## Objective

Train and compare a Decision Tree Classifier and a Random Forest Classifier to predict whether a flight will be delayed or depart on time.

## Tasks

1. Explore the script: Open `compare_models.py` and identify the input features (`Hour`, `Weather`, and `Congestion`) and the target variable (`Delayed`).
2. Run the models: Execute the script to train both classifiers and generate predictions on the test dataset.
3. Compare the results: Record the accuracy of each model and identify which achieved higher accuracy on the test set.
4. Update the Python script as follows: line `8` contains a variable called `RANDOM_STATE`. This allows the models to preserve state in that the results are predictable (same results on multiple runs). Modify this variable with your university ID and re-run the script.
5. Compare the results obtained aftering modifying the file. The comparison is saved automatically in a `comparison.txt` file.

## Deliverables

Upload the following file to the Moodle assignment:

* The Python script (`compare_models.py`), including your modifications.
* The prediction file (`comparison.txt`).

Include a brief comparison of the results in your submission, indicating the accuracy of both models and your main conclusions.

Important: This is an educational demonstration using a small, synthetic dataset. The evaluation results are intended to illustrate model comparison and must not be interpreted as evidence of reliable real-world flight-delay prediction.

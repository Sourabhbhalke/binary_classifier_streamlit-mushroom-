# binary_classifier_streamlit-mushroom-web-app
A web app for classifying mushrooms as edible or poisonous using machine learning algorithms: Support Vector Machine (SVM), Logistic Regression, and Random Forest


This web application is designed to classify mushrooms as either edible or poisonous based on several features. It utilizes three different machine learning algorithms: **Support Vector Machine (SVM)**, **Logistic Regression**, and **Random Forest**. The app provides an interactive interface to tune hyperparameters, visualize model performance, and interpret results through various plots.

## Algorithms and Hyperparameters

### 1. **Support Vector Machine (SVM)**
   - **C (Regularization Parameter):** Controls the trade-off between achieving a low error on the training data and minimizing the model complexity.
   - **Kernel:** Determines the type of decision boundary used by the SVM. Available options:
     - `linear`: A linear decision boundary.
     - `rbf`: A radial basis function (Gaussian) kernel.
     - `poly`: A polynomial kernel.
     - `sigmoid`: A sigmoid kernel.
   - **Gamma:** Defines how far the influence of a single training example reaches, with options:
     - `scale`: Default value, which uses `1 / (n_features * X.var())`.
     - `auto`: Uses `1 / n_features`.

### 2. **Logistic Regression**
   - **C (Regularization Parameter):** Inverse of regularization strength. Smaller values specify stronger regularization.
   - **Max Iterations:** Specifies the maximum number of iterations for the solver to converge.

### 3. **Random Forest**
   - **Number of Trees (n_estimators):** The number of decision trees in the forest.
   - **Maximum Depth:** The maximum depth of the trees. Controls overfitting by limiting how deep the trees can grow.

## Performance Metrics and Plots

### 1. **Confusion Matrix**
   - A table that allows visualization of the performance of the model by showing the true positive, true negative, false positive, and false negative predictions.

### 2. **ROC Curve (Receiver Operating Characteristic)**
   - A graphical plot that illustrates the diagnostic ability of a binary classifier. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

### 3. **Precision-Recall Curve**
   - A plot that shows the trade-off between precision (the accuracy of the positive predictions) and recall (the ability of the model to find all the positive samples). This is particularly useful in scenarios with imbalanced datasets.

## Use-cases Demonstrated
- **Hyperparameter Tuning:** The app allows users to adjust key hyperparameters for each algorithm to understand their impact on model performance.
- **Model Evaluation:** The app provides various metrics and plots to evaluate and compare the performance of different algorithms, to get an understanding of model diagnostics.
- **Visualization:** The use of interactive plots like Confusion Matrix, ROC Curve, and Precision-Recall Curve will provide help in interpreting model results and conveying insights effectively.


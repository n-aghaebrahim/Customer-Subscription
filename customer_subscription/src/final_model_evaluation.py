from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define a function for evaluating the model
def evaluate_model(model, x_test, y_test):
    
    # Predict the target values using the trained model
    y_pred = model.predict(x_test)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Print evaluation metrics
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1))
    print("Confusion Matrix:")
    print("TP: {}, FP: {}, TN: {}, FN: {}".format(tp, fp, tn, fn))
    
# Evaluate the final models on the test set
print("Random Forest:")
evaluate_model(rf_best, x_test, y_test)
print("\nSVM:")
evaluate_model(svm_best, x_test, y_test)
print("\nXGBoost:")
evaluate_model(xgb_best, x_test, y_test)
print("\nDecision Tree:")
evaluate_model(dt_best, x_test, y_test)


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# Define models
rfc = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
xgb = XGBClassifier(random_state=42)
dtc = DecisionTreeClassifier(random_state=42)

models = [rfc, svc, xgb, dtc]

# Define number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Loop over models
for model in models:
    print(f"Model: {model.__class__.__name__}")
    acc_scores = []
    # Loop over folds
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        # Fit model on training data
        model.fit(X_tr, y_tr)
        # Predict on validation data
        y_pred = model.predict(X_val)
        # Calculate accuracy score
        acc_score = accuracy_score(y_val, y_pred)
        acc_scores.append(acc_score)
    # Calculate and print mean accuracy score
    mean_acc_score = sum(acc_scores) / n_splits
    print(f"Mean accuracy score: {mean_acc_score:.3f}\n")


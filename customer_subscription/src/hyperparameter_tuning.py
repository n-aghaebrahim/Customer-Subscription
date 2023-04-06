from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Random Forest hyperparameter search range
rf_param_dist = {'n_estimators': randint(50, 500),
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'max_depth': [10, 20, 30, 40, 50, None],
                 'min_samples_split': randint(2, 10),
                 'min_samples_leaf': randint(1, 10)}

# SVM hyperparameter search range
svm_param_dist = {'C': uniform(0.1, 10),
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  'degree': randint(1, 10)}

# XGBoost hyperparameter search range
xgb_param_dist = {'learning_rate': uniform(0.01, 0.5),
                  'n_estimators': randint(50, 500),
                  'max_depth': randint(1, 10),
                  'min_child_weight': randint(1, 10),
                  'gamma': uniform(0, 1),
                  'subsample': uniform(0.1, 1),
                  'colsample_bytree': uniform(0.1, 1)}

# Decision Tree hyperparameter search range
dt_param_dist = {'criterion': ['gini', 'entropy'],
                 'max_depth': [10, 20, 30, 40, 50, None],
                 'min_samples_split': randint(2, 10),
                 'min_samples_leaf': randint(1, 10)}

# Random Forest hyperparameter tuning
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_dist, n_iter=100, cv=3, random_state=42)
rf_random.fit(X_train, y_train)
rf_best_params = rf_random.best_params_

# SVM hyperparameter tuning
svm = SVC()
svm_random = RandomizedSearchCV(estimator=svm, param_distributions=svm_param_dist, n_iter=100, cv=3, random_state=42)
svm_random.fit(X_train, y_train)
svm_best_params = svm_random.best_params_

# XGBoost hyperparameter tuning
xgb = XGBClassifier()
xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_dist, n_iter=100, cv=3, random_state=42)
xgb_random.fit(X_train, y_train)
xgb_best_params = xgb_random.best_params_

# Decision Tree hyperparameter tuning
dt = DecisionTreeClassifier()
dt_random = RandomizedSearchCV(estimator=dt, param_distributions=dt_param_dist, n_iter=100, cv=3, random_state=42)
dt_random.fit(X_train, y_train)
dt_best_params = dt_random.best_params_


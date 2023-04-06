from xgboost import XGBClassifier

# XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)

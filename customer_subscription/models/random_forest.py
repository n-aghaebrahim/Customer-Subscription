from sklearn.ensemble import RandomForestClassifier


# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)


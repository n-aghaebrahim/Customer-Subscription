from sklearn.tree import DecisionTreeClassifier

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)

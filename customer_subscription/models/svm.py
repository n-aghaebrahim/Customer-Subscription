from sklearn.svm import SVC

# SVM
svm = SVC(C=1, kernel='rbf', gamma='auto')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)

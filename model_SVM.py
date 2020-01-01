""" this file defines an svm classifier """
from sklearn.svm import SVC
from preprocess_data import return_when_called
from sklearn.metrics import mean_absolute_error,classification_report

#svm classifier with gaussian kernel
svclassifier = SVC(kernel='rbf') 

#gets the X and Y arrays from preprocess_data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_when_called()

#fitting the model
svclassifier.fit(X_train_scaled,y_train)

#making predictions on validation set
val_predictions = svclassifier.predict(X_val_scaled)

#calculating and printing mean absolute error
mae = mean_absolute_error(val_predictions,y_val)
print("The mean absolute error is: " + str(mae))

#making predictions on test data and printing classification report
test_predictions = svclassifier.predict(X_test_scaled)
print(classification_report(y_test,test_predictions))
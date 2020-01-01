""" this file implememts a logistic regression classifier """
from sklearn.linear_model import LogisticRegression
from preprocess_data import return_when_called
from sklearn.metrics import mean_absolute_error,classification_report

#gets the X and Y arrays from preprocess_data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_when_called()

#instance of the class
logisticregression = LogisticRegression()

#fitting the model
logisticregression.fit(X_train_scaled,y_train)

#making predictions on validation set
val_predictions = logisticregression.predict(X_val_scaled)

#mae calculation
mae = mean_absolute_error(y_val,val_predictions)
print("The mean absolute error is: " + str(mae))

#making predictions on test set
test_predictions = logisticregression.predict(X_test_scaled)
print(classification_report(y_test,test_predictions))
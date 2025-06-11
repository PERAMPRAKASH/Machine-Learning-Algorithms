from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
iris=datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model= LinearRegression()
model.fit(x_train,y_train)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Score:", model.score(x_test, y_test))
print("Predictions:", model.predict(x_test))
import numpy as np 
import matplotlib.pyplot as plt
# Visualizing the predictions
plt.scatter(y_test, model.predict(x_test))
plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.axis('tight')   
plt.show()
plt.savefig("linear_regression_plot.png")   
# Save the model
import joblib
joblib.dump(model, 'linear_regression_model.pkl')
# Load the model
loaded_model = joblib.load('linear_regression_model.pkl')
# Verify the loaded model
print("Loaded Model Coefficients:", loaded_model.coef_)
print("Loaded Model Intercept:", loaded_model.intercept_)
print("Loaded Model Score:", loaded_model.score(x_test, y_test))
# Make predictions with the loaded model
predictions = loaded_model.predict(x_test)
print("Loaded Model Predictions:", predictions)
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


*OUTPUT:*
Coefficients: [-0.11633479 -0.05977785  0.25491375  0.54759598]
Intercept: 0.2525275898181477
Score: 0.9468960016420045
Predictions: [ 1.23071715 -0.04010441  2.21970287  1.34966889  1.28429336  0.02248402
  1.05726124  1.82403704  1.36824643  1.06766437  1.70031437 -0.07357413
 -0.15562919 -0.06569402 -0.02128628  1.39659966  2.00022876  1.04812731
  1.28102792  1.97283506  0.03184612  1.59830192  0.09450931  1.91807547
  1.83296682  1.87877315  1.78781234  2.03362373  0.03594506  0.02619043]
Loaded Model Coefficients: [-0.11633479 -0.05977785  0.25491375  0.54759598]
Loaded Model Intercept: 0.2525275898181477
Loaded Model Score: 0.9468960016420045
Loaded Model Predictions: [ 1.23071715 -0.04010441  2.21970287  1.34966889  1.28429336  0.02248402
  1.05726124  1.82403704  1.36824643  1.06766437  1.70031437 -0.07357413
 -0.15562919 -0.06569402 -0.02128628  1.39659966  2.00022876  1.04812731
  1.28102792  1.97283506  0.03184612  1.59830192  0.09450931  1.91807547
  1.83296682  1.87877315  1.78781234  2.03362373  0.03594506  0.02619043]

import numpy as np
import pandas as pd
class Logistic_Regression():
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
    def fit(self, X, Y):
        self.m,self.n = X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y
        for i in range(self.n_iterations):
            self.update_weights()
            
    def update_weights(self):       
        Y_hat = 1/(1+np.exp(-(self.X.dot(self.w) + self.b)))
        dw=(1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))
        db=(1/self.m)*np.sum(Y_hat - self.Y)
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        
    def predict(self, X):
        Y_pred = 1/(1+np.exp(-(X.dot(self.w) + self.b)))
        Y_pred = [1 if i > 0.5 else 0 for i in Y_pred]
        return np.array(Y_pred)
    
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('diabetes.csv')
X=df.drop('Outcome',axis=1).values
Y=df['Outcome'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X = np.vstack((X_train, X_test))
Y = np.hstack((Y_train, Y_test))
model = Logistic_Regression(learning_rate=0.01, n_iterations=1000)
model.fit(X, Y)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Model Weights: {model.w}")

print(f"Model Bias: {model.b}")
print(f"Predictions: {Y_pred}")


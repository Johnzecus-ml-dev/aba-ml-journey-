from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib

X, y = load_diabetes(return_X_y=True)
model = LinearRegression().fit(X, y)
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")

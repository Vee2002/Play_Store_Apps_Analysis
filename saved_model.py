
import joblib
from sklearn.metrics import precision_score

# Load the variables and model
X_train_encoded = joblib.load('X_train_encoded.pkl')
X_test_encoded = joblib.load('X_test_encoded.pkl')
y_train_encoded = joblib.load('y_train_encoded.pkl')
y_test_encoded = joblib.load('y_test_encoded.pkl')
best_model = joblib.load('best_model.pkl')

# Make predictions
prediction = best_model.predict(X_test_encoded)

# Evaluate and print the precision score
print(f"Precision score: {precision_score(y_test_encoded, prediction, average='weighted')}")


from flask import Flask, request, render_template
import joblib
import numpy as np
from saved_model import best_model 

app = Flask(__name__)

# Example route for the home page
@app.route('/')
def home():
    return render_template('deployment.html')

# Example route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting the form data (assuming form fields are named 'feature1', 'feature2', 'feature3')
        feature1 = float(request.form['Ratings'])
        feature2 = float(request.form['Size(KB)'])
        feature3 = float(request.form['Reviews'])

        # Create an input array for prediction
        input_features = np.array([[feature1, feature2, feature3]])

        # Making prediction using the model
        prediction = best_model.predict(input_features)

        # Return the prediction as a response
        return f"The prediction is: {prediction[0]}"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import pickle
import os

# Load the pre-trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Route for HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction from HTML form
@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        # Get input from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare input features
        features = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Predict species
        prediction = model.predict(features)[0]
        species = {0: "setosa", 1: "versicolor", 2: "virginica"}

        return render_template('index.html', prediction=species[prediction])
    except Exception as e:
        return render_template('index.html', error="Invalid input. Please try again.")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

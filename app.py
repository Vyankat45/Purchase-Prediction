from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load pipeline model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        # Create a DataFrame (since our pipeline expects columns)
        input_data = pd.DataFrame([[gender, age, salary]], columns=['Gender', 'Age', 'Salary'])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Will Purchase ✅" if prediction == 1 else "Will Not Purchase ❌"

        return render_template('index.html', prediction_text=f"Prediction: {result}")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

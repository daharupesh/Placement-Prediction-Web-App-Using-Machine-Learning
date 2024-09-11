from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model from the pickle file
model = pickle.load(open('model.pkl', 'rb'))

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    iq = float(request.form['iq'])
    cgpa = float(request.form['cgpa'])

    # Prepare the input for the model
    features = np.array([[cgpa, iq]])

    # Make the prediction using the model
    prediction = model.predict(features)

    # Convert prediction to a human-readable form
    output = 'placement will be' if prediction[0] == 1 else 'placement will not be'

    return render_template('index.html', prediction_text=f'Placement: {output}')

if __name__ == "__main__":
    app.run(debug=True)

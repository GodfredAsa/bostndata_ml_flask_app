import numpy as np

from flask import Flask, request, jsonify
import joblib

model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert the data into a numpy array (assume data is a list of feature values)
    features = np.array([data['features']])

    if len(features[0]) <= 14:
        error_response = {'error': True, 'status': 400, 'data': None, 'message': "Number of floats must be 15"}
        print(error_response)
        return error_response, 400

    # Make a prediction using the model
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    predicted_result = float(prediction[0])
    retJson = {
        'error': 'False', 'status': 200,
        'data': {
            'prediction': predicted_result,
            'price': round(predicted_result * 1000, 3),
            'currency': 'USD',
            "multiplier": 1000},
        'message': 'Prediction Successful'
    }
    print(retJson)
    return jsonify(retJson), 200


if __name__ == '__main__':
    # Run the app on the local development server
    app.run(debug=True, host='0.0.0.0', port=5001)

#     curl -X POST -H "Content-Type: application/json" -d '{"features": [0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.9, 4.98, 4, 1]}' http://127.0.0.1:5000/predict

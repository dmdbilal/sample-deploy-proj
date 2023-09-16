from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Sample model training data
np.random.seed(0)
X = np.random.rand(100, 1) * 5
y = 2 * X + 1 + np.random.randn(100, 1)
model = LinearRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        bedrooms = data.get('bedrooms')

        if bedrooms is None:
            return jsonify({'error': 'Please provide the number of bedrooms.'}), 400

        # Ensure the input is a float
        bedrooms = float(bedrooms)

        # Make a prediction using the trained model
        prediction = model.predict([[bedrooms]])

        return jsonify({'predicted_price': prediction[0][0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

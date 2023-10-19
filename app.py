# importing flask module
import pickle

import numpy as np
from flask import Flask, request, jsonify

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


# creating routes
@app.route('/')
def home():
    return "Hello World here"


# api will exist in this route
# api will handle post request, there are 2 types of request get and post
# in get, we take input through url... In post without url

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    # forming dictionary of result
    # result = {'cgpa': cgpa, 'iq': iq, 'profile_score': profile_score}

    # this is input
    input_query = np.array([[cgpa, iq, profile_score]], dtype=np.float64)

    # this is output will be in the form of array
    result = model.predict(input_query)[0]

    # returning in the form of json
    return jsonify({'placement': str(result)})


# Postman jo h vo http request mar rha h.. and flask api is returning the dictioning in form of json

if __name__ == '__main__':
    app.run(debug=True)


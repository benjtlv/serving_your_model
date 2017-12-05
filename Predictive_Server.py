from flask import Flask
from flask import jsonify
from flask import request
from pickle import load
from nltk import ngrams
import numpy as np
app = Flask(__name__)

predictive_model = load(open('logistic.sav', 'rb'))
features = load(open('features.sav', 'rb'))


def ngrams_vector(baby_name):
    """returns a vector understandable by the model"""
    baby_name = baby_name.lower()
    bigrams = ngrams(baby_name, 2)
    bigrams = list(map(lambda tuple: str(tuple[0]) + str(tuple[1]), bigrams))
    vector = list(map(lambda e: 1.0 if e in bigrams else 0.0, features))
    return np.array([vector])


@app.route('/baby/<string:baby_name>')
def predict_baby_sex(baby_name):
    x = ngrams_vector(baby_name)
    y = predictive_model.predict(x)
    return '%s is a girl : %i' % (baby_name, y, )


@app.route('/baby/list_gender', methods=['POST'])
def predict_list_gender():
    _,  list_names = request.form.items()
    result = []
    for name in list_names:
        vect = ngrams_vector(name)
        result.append(predictive_model.predict(vect))
    return jsonify(result)

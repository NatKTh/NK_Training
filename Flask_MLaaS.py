import joblib
import numpy as np
from flask import Flask,request
from flask_cors import CORS,cross_origin

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello NK Postman World!."


@app.route('/iris',methods=['POST'])
@cross_origin()
def predict_species():
    model = joblib.load('iris.model')
    req = request.values['param']
    inputs = np.array(req.split(','),dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'Setosa'
    elif predict_target == 1:
        return 'Versicolour'
    else:
        return 'Virginica'

app.run(host='0.0.0.0',port=5000,debug=False)
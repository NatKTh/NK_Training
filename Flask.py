from flask import Flask , request
from flask_cors import CORS , cross_origin

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Hello NK Web Again!"

@app.route('/area',methods=['GET'])
@cross_origin()
def area():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w*h)

@app.route('/bmi',methods=['GET'])
@cross_origin()
def bmi():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w/(h*h))

app.run(host='0.0.0.0',port=5000,debug=False)
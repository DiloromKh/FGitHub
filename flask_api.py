from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    Age=request.args.get("Age")
    EDUC=request.args.get("EDUC")
    eTIV=request.args.get("eTIV")
    nWBV=request.args.get("nWBV")
    ASF=request.args.get("ASF")
    CDR=request.args.get("CDR")
    
    prediction=classifier.predict([[Age,EDUC,eTIV,nWBV,ASF,CDR]])
    print(prediction)
    return "Hello The answer is"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
  
    data_test=pd.read_csv(request.files.get("file"))
    print(data_test.head())
    prediction=classifier.predict(data_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8501)




    

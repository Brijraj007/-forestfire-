import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler #To load 'pickle file'

app = Flask(__name__)

##import ridge regressor model & standard scaler Pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Rodute for home page
@app.route('/')  
def index():
    return render_template('index.html') #render_template: see template folder
#Predict data
@app.route('/predictdata',methods=['GET','POST']) #yah page sabhi input likhenga  #Html WebDevloper bana ke denge; methods=['GET','POST']:to handle both; #Understand: 'GET'-agar yaha home.html direct dissplay kar raha hu, 'POST-agar form ko submit kar raha hu
def predict_datapoint(): #predict_datapoint : from [home.html [line 7]], #Now, .html ke sabhi input_value ko read karunga
    if request.method=='POST':  #request : specific method ko access kar pata hai, #[.html [line 7 ]] 'POST' will hit
# """Remember: jaise model ko prediction,training karate wakta columns order tha, see: input Data==Below Code usi order me ho"""
        Temperature=float(request.form.get('Temperature')) #'Temperature' match with '.html'[line 8: name="Temperature"]
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
#"""Scaling"""
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
               #"""Ridge"""
        result=ridge_model.predict(new_data_scaled) #display 'result' in [home.html[line 23]] isse pass karne ke liye see [.py: line 36]

        return render_template('home.html',result=result[0]) #result[0] : [0] bcz. yah list me hai aur 1st element access karna hai

    else:
        return render_template('home.html')  #'GET'- this is 'GET' functionality



if __name__=="__main__":
    app.run(host="0.0.0.0") #host="0.0.0.0" : jaha bhi run karega yah localAddress ke saath match karlega
#in Flask by default port run in :5000; you can give your specific port by : port='jo port available hai in Machine'



#URL for this page:
#https://brown-psychiatrist-ndfjr.pwskills.app:5000/predictdata  #-predictdataa



###Check mdels/scalers,modls/85regressors why my model show error
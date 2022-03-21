from pathlib import Path
import joblib as jb
from flask import Flask, render_template, request
import numpy as np



app= Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict(): 
    ph=(request.form['ph'])
    Hardness=(request.form['Hardness'])
    Solids=(request.form['Solids'])
    Chloramines=(request.form['Chloramines'])
    Sulfate=(request.form['Sulfate'])
    Conductivity=(request.form['Conductivity'])
    Organic_carbon=(request.form['Organic_carbon'])
    Trihalomethanes=(request.form['Trihalomethanes'])
    Turbidity=(request.form['Turbidity'])
    
    arr=np.array([[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]])
   
    model=jb.load(Path('model\model_rfc.pkl'))
    result= model.predict(arr)
    if result ==1 :
        return render_template('after.html',data= 'Its safe drink')
    else:
        return render_template('after.html',data= 'not a drinking water')
if __name__ == '__main__':
    app.run()
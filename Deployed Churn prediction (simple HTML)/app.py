import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('/Users/apple/Desktop/Demo versions for Paymona website/three_notebooks/churn/model_test.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('deploy_test.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print((request.form))
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = [ 'gender', 'Partner', 'Dependents', 
    'tenure','PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity','OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 
    'PaymentMethod', 'MonthlyCharges','TotalCharges'])                         
    
    prediction = model.predict(data_unseen)

    output = prediction
    
    if output == 1:
        label = 'Churn'
    else:
        label = 'Not Churn'

    return render_template('deploy_test.html', prediction_text=label)

if __name__ == "__main__":
    app.run(debug=True)
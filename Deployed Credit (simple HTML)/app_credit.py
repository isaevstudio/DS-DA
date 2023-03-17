import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from keras.models import load_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('/Users/apple/Desktop/Demo versions for Paymona website/three_notebooks/credit1/rdf_model.pkl', 'rb'))
# model_eda = pickle.load(open('/Users/apple/Desktop/Demo versions for Paymona website/three_notebooks/credit1/pipeline_newColumn.pkl', 'rb'))


def TotalNumberOfTime_df(df):
    df['TotalNumberOfTime'] = df['NumberOfTime30-59DaysPastDueNotWorse'] + df['NumberOfTime60-89DaysPastDueNotWorse'] + df['NumberOfTimes90DaysLate']
    df['TotalNumberOfOpenCredit'] = df['NumberOfOpenCreditLinesAndLoans'] + df['NumberRealEstateLoansOrLines']
    return df

def IsDefaulted_df(df):
    df['IsDefaulted'] = int(df['NumberOfTimes90DaysLate']) + int(df['NumberOfTime60-89DaysPastDueNotWorse']) + int(df['NumberOfTime30-59DaysPastDueNotWorse'])
    df['IsDefaulted'] = int(df['IsDefaulted'])
    df.loc[(df["IsDefaulted"] >= 1), "IsDefaulted"] = 1
    return df

TotalNumberOfTime = FunctionTransformer(TotalNumberOfTime_df,validate=False)
IsDefaulted = FunctionTransformer(IsDefaulted_df,validate=False)

pipeline_newColumns = Pipeline([('step2', TotalNumberOfTime),
                                ('step3', IsDefaulted)])   





@app.route('/')
def home():
    return render_template('deploy_credit.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print((request.form))
    int_features = [x for x in request.form.values()]
    int_features[0] = int(int_features[0])/100
    int_features[3] = int(int_features[3])/100
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = [ 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents'])  

    # data_unseen = pd.DataFrame((final[0], final[1],final[2],final[3],final[4],final[5], final[6], final[7], final[8],
    # final[9],final[10]), columns = [ 'RevolvingUtilizationOfUnsecuredLines', 'age',
    # 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    # 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    # 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    # 'NumberOfDependents'])                         
    
    # preprediction = model_eda.fit_transform(data_unseen)
    # prediction = model.predict(preprediction)
    prediction = model.predict(pipeline_newColumns.fit_transform(data_unseen))
    output = prediction
    
    if output == 1:
        label = 'Rejected'
    else:
        label = 'Agreed'

    return render_template('deploy_credit.html', prediction_text=label)

if __name__ == "__main__":
    app.run(debug=True)
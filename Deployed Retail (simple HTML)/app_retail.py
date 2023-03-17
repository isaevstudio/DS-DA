from flask.helpers import stream_with_context
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect
import pickle
from keras.models import load_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


app = Flask(__name__)  # Initialize the flask App
# df_converter = pickle.load(open('/Users/apple/Desktop/Demo versions for Paymona website/three_notebooks/retail/edtion1/df_converter.pkl', 'rb'))
model = pickle.load(open(
    '/Users/apple/Desktop/Demo versions for Paymona website/three_notebooks/retail/edtion1/model_test1.pkl', 'rb'))
daily_avg = pd.read_csv(
    '/Users/apple/Desktop/Demo versions for Paymona website/three_notebooks/retail/edtion1/daily_avg.csv')
monthly_avg = pd.read_csv(
    '/Users/apple/Desktop/Demo versions for Paymona website/three_notebooks/retail/edtion1/monthly_avg.csv')


@app.route('/')
def home():
    return render_template('deploy_retail.html')


@app.route('/', methods=['POST'])
def home_post():

    start_date = request.form.get('start')
    end_date = request.form.get('end')
    stores = request.form.getlist('store')
    products = request.form.getlist('product')

    # preparing Dates; Stores and products are already ready for use
    dates = pd.date_range(start=start_date, end=end_date)

    # Creating two temporary lists that will store the lists of the Stores&Products, Dates&Stores&Products
    temp_StoreProduct = []
    temp_DateStoreProduct = []
    for store in stores:
        for product in products:
            temp_StoreProduct.append([store, product])

    for date in dates:
        for StPr in temp_StoreProduct:
            temp_DateStoreProduct.append([date, StPr[0], StPr[1]])

    # Converting the merged lists to data frame
    dataframe = pd.DataFrame(temp_DateStoreProduct, columns=[
        'date', 'store', 'item'])
    dataframe['store'] = dataframe['store'].astype('int')
    dataframe['item'] = dataframe['item'].astype('int')

    def date_features(df):
        # Date Features
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df.date.dt.year
        df['month'] = df.date.dt.month
        df['day'] = df.date.dt.day
        df['dayofyear'] = df.date.dt.dayofyear
        df['dayofweek'] = df.date.dt.dayofweek
        df['weekofyear'] = df.date.dt.weekofyear

        df.drop('date', axis=1, inplace=True)

        return df

    def merge(df1, df2, col, col_name):

        df1 = pd.merge(df1, df2, how='left', on=None, left_on=col, right_on=col,
                       left_index=False, right_index=False, sort=True,
                       copy=True, indicator=False)

        df1 = df1.rename(columns={'sales': col_name})
        return df1

    dataframe = date_features(dataframe)

    dataframe = merge(dataframe, daily_avg, [
                      'item', 'store', 'dayofweek'], 'daily_avg')
    dataframe = merge(dataframe, monthly_avg, [
                      'item', 'store', 'month'], 'monthly_avg')

    # prediction = model.predict(pipeline_newColumns.fit_transform(dataframe))
    prediction = model.predict(dataframe)
    dataframe = dataframe.drop(columns=[
        'dayofweek', 'weekofyear', 'daily_avg', 'monthly_avg', 'dayofyear'])
        

    a = pd.DataFrame(np.round(prediction), columns=['Sales'])
    table_prediction = dataframe.join(a)

    # html_table_head = table_prediction.head().to_html()
    html_table_full = table_prediction.to_html()

    return render_template('deploy_retail.html', pred_results=html_table_full)


if __name__ == "__main__":
    app.run(debug=True)

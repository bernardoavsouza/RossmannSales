# -*- coding: utf-8 -*-
import pickle
import pandas as pd
from flask import Flask, request, Response
from classes.pipeline import Rossmann


model = pickle.load(open(r'C:\Users\Noudy\Desktop\Estudo\rossmann-sales\models\final_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
   
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
    
        # Instantiate Rossmann class
        pipeline = Rossmann()
        
        df1 = pipeline.data_cleaning( test_raw )
        df2 = pipeline.feature_engineering( df1 )
        df3 = pipeline.data_preparation( df2 )
    
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
    
    else:    
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    app.run('127.0.0.1')

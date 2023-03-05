# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import pickle



class Rossmann():
    def __init__(self):
        self.path = r'C:/Users/Noudy/Desktop/Estudo/rossmann-sales'
        self.StoreType_encoding = pickle.load(open(self.path + '/parameter/StoreType_encoding.pkl', 'rb'))
        self.CompetitionDistance_scaling = pickle.load(open(self.path + '/parameter/CompetitionDistance_scaling.pkl', 'rb'))
        self.CompetitionDuration_scaling = pickle.load(open(self.path + '/parameter/CompetitionDuration_scaling.pkl', 'rb'))
        self.PromoDuration_scaling = pickle.load(open(self.path + '/parameter/PromoDuration_scaling.pkl', 'rb'))
        self.Year_scaling = pickle.load(open(self.path + '/parameter/Year_scaling.pkl', 'rb'))
    
    def data_cleaning(self, df1):
        
        df1.columns = ['Store', 'DayOfWeek', 'Date', 
                       'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                       'StoreType', 'Assortment', 'CompetitionDistance',
                       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                       'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 
                       'PromoInterval'                                         ]

       
        ##  Data Types
        df1['Date'] = pd.to_datetime(df1['Date'])
        
 
        ## Fillout NA

        # CompetitionDistance    
        df1['CompetitionDistance'] = df1['CompetitionDistance'].fillna(250000)

        # CompetitionOpenSinceMonth 
        df1['CompetitionOpenSinceMonth'] = df1.apply(lambda x: x['Date'].month if np.isnan(x['CompetitionOpenSinceMonth']) else x['CompetitionOpenSinceMonth'], axis = 1)

        # CompetitionOpenSinceYear   
        df1['CompetitionOpenSinceYear'] = df1.apply(lambda x: x['Date'].year if np.isnan(x['CompetitionOpenSinceYear']) else x['CompetitionOpenSinceYear'], axis = 1)

        # Promo2SinceWeek      
        df1['Promo2SinceWeek'] = df1.apply(lambda x: x['Date'].weekofyear if np.isnan(x['Promo2SinceWeek']) else x['Promo2SinceWeek'], axis = 1)

        # Promo2SinceYear              
        df1['Promo2SinceYear'] = df1.apply(lambda x: x['Date'].year if np.isnan(x['Promo2SinceYear']) else x['Promo2SinceYear'], axis = 1)

        # PromoInterval                
        df1['PromoInterval'] = df1['PromoInterval'].fillna(0)

        ## Change Data Types

        # CompetitionOpenSinceMonth  
        df1['CompetitionOpenSinceMonth'] = df1['CompetitionOpenSinceMonth'].astype(np.int64)

        # CompetitionOpenSinceYear            
        df1['CompetitionOpenSinceYear'] = df1['CompetitionOpenSinceYear'].astype(np.int64)

        # Promo2SinceWeek                   
        df1['Promo2SinceWeek'] = df1['Promo2SinceWeek'].astype(np.int64)

        # Promo2SinceYear                   
        df1['Promo2SinceYear'] = df1['Promo2SinceYear'].astype(np.int64)


        MonthDict = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['MonthDict'] = df1['Date'].dt.month.map(MonthDict)

        df1['IsPromo'] = df1[['PromoInterval', 'MonthDict']].apply(lambda x: 0 if x['PromoInterval'] == 0 else 1 if x['MonthDict'] in x['PromoInterval'].split(',') else 0, axis=1)

        df1['StateHoliday'] = df1['StateHoliday'].apply(lambda x: 0 if x == '0' else x)
        
        
        return df1
    
    
    def feature_engineering(self, df2):
        
        # DateTimes
        df2['Day'] = df2['Date'].dt.day
        df2['Month'] = df2['Date'].dt.month
        df2['Year'] = df2['Date'].dt.year
        df2['WeekOfYear'] = df2['Date'].dt.weekofyear

        df2['CompetitionSince'] = df2.apply(lambda x: datetime.datetime(year = x['CompetitionOpenSinceYear'], month = x['CompetitionOpenSinceMonth'], day = 1), axis = 1)
        df2["CompetitionDurationMonth"] = df2.apply(lambda x: (x['Date'] - x['CompetitionSince'])/30, axis = 1).apply(lambda x: x.days)

        df2['PromoSince'] = df2.apply(lambda x: datetime.date.fromisocalendar(x['Promo2SinceYear'], x['Promo2SinceWeek'], 1), axis = 1)
        df2['PromoSince'] = pd.to_datetime(df2['PromoSince'])
        df2['PromoDurationWeek'] = df2.apply(lambda x: (x["Date"] - x['PromoSince']) / 7, axis = 1).apply(lambda x: x.days)

        # State Holiday
        holidays_dict = {'a': 'Public', 'b': 'Easter', 'c': 'Christmas', 0: 'Regular'}
        df2['StateHoliday'] = df2['StateHoliday'].map(holidays_dict)

        # Assortment
        assortment_dict = {'a': 'Basic', 'b': 'Extra', 'c': 'Extended'}
        df2['Assortment'] = df2['Assortment'].map(assortment_dict)
    
        df2 = df2[df2["Open"] == 1]

        ## Columns Filtering

        cols2drop = ['MonthDict', 'PromoInterval', 'Open']
        df2.drop(cols2drop, axis = 1, inplace = True)
        
        return df2
    
    
    def data_preparation(self, df5):
        
        # StateHoliday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['StateHoliday'], columns=['StateHoliday'])

        # StoreType - Label Encoding
        le = self.StoreType_encoding
        df5['StoreType'] = le.transform(df5['StoreType'])


        # Assortment - Ordinal Encoding
        assortment_dict = {'Basic': 1, 'Extra': 2, 'Extended': 3}
        df5['Assortment'] = df5['Assortment'].map( assortment_dict)


        ### Nature Transformation

        # Day Of Week
        df5['DayOfWeek_sin'] = df5['DayOfWeek'].apply(lambda x: np.sin(x * (2. * np.pi/7)))
        df5['DayOfWeek_cos'] = df5['DayOfWeek'].apply(lambda x: np.cos(x * (2. * np.pi/7)))

        # Month
        df5['Month_sin'] = df5['Month'].apply(lambda x: np.sin(x * (2. * np.pi/12)))
        df5['Month_cos'] = df5['Month'].apply(lambda x: np.cos(x * (2. * np.pi/12)))

        # Day
        df5['Day_sin'] = df5['Day'].apply(lambda x: np.sin(x * (2. * np.pi/30)))
        df5['Day_cos'] = df5['Day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))

        # Week Of Year
        df5['WeekOfYear_sin'] = df5['WeekOfYear'].apply(lambda x: np.sin(x * (2. * np.pi/52)))
        df5['WeekOfYear_cos'] = df5['WeekOfYear'].apply(lambda x: np.cos(x * (2. * np.pi/52)))


        ## Scaling

        # Competition Distance
        rs = self.CompetitionDistance_scaling
        df5['CompetitionDistance'] = rs.transform(df5[['CompetitionDistance']].values)


        # Competition Duration Month
        rs = self.CompetitionDuration_scaling
        df5['CompetitionDurationMonth'] = rs.transform(df5[['CompetitionDurationMonth']].values)


        # Promo Duration Week
        mms = self.PromoDuration_scaling
        df5['PromoDurationWeek'] = mms.transform(df5[['PromoDurationWeek']].values)


        # Year
        mms = self.Year_scaling
        df5['Year'] = mms.transform(df5[['Year']].values)
        
        cols_selected = ['Store', 'Promo', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                         'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'CompetitionDurationMonth',
                         'PromoDurationWeek', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
                         'WeekOfYear_sin', 'WeekOfYear_cos']
        
        df5 = df5[cols_selected]
        
        return df5
    
    
    def get_prediction(self, model, raw_data, test_data):
        
        pred = model.predict(test_data)
        
        raw_data['Predictions'] = np.expm1(pred)
        
        return raw_data.to_json(orient = 'records', date_format = 'iso')
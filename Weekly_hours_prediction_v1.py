import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn import linear_model, metrics 
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from itertools import product
from flask import Flask, request, jsonify
#app = Flask(__name__)
# Need to set a local environment variable for running the app locally with the demo UI
#LOCALHOST = os.environ.get('FLASK_LOCAL')

#if LOCALHOST:
from flask_cors import CORS
#CORS(app)
class ReportService():
    def __init__(self):
        #self.dataset = pd.read_excel(io.BytesIO(uploaded['incident.xlsx']))
        self.dataset = pd.read_excel('./incident.xlsx')
        self._clean_incident_dataset()
        #return self.dataset_week
        
    def _clean_incident_dataset(self):
        self.dataset = self.dataset[self.dataset.resolved_at.notnull()]
        self.dataset['week'] = self.dataset['resolved_at'].dt.week
        self.dataset['year'] = self.dataset['opened_at'].dt.year
        self.year = self.dataset['year'].max()
        self.incident_bd = pd.DataFrame(self.dataset.business_duration)
        z = np.abs(stats.zscore(self.incident_bd))
        threshold = 3
        Q1 = self.dataset.quantile(0.25)
        Q3 = self.dataset.quantile(0.75)
        IQR = Q3 - Q1
        self.dataset = self.dataset[(z < 3).all(axis=1)] # removed all the records which is 3 std far
        #dataset.shape
        self.dataset['category'] = self.dataset['category'].fillna('Other')
        le = LabelEncoder()
        self.dataset['category']= le.fit_transform(self.dataset['category']).astype("uint8")
        self.le_cat_mapping = dict(zip(le.transform(le.classes_),le.classes_))
        self.dataset['priority']= le.fit_transform(self.dataset['priority']).astype("uint8")
        self.le_pri_mapping = dict(zip(le.transform(le.classes_),le.classes_))
        
    def _preapre_incident_dataset(self):
        # Create "grid" with columns
        index_cols = ['category', 'priority', 'week']
        # For every week we create a grid from all category and priority combinations from that week
        grid = []
        for week in self.dataset['week'].unique():
            self.cur_cat = self.dataset.loc[self.dataset['week'] == week, 'category'].unique()
            self.cur_pri = self.dataset.loc[self.dataset['week'] == week, 'priority'].unique()
            grid.append(np.array(list(product(*[self.cur_cat, self.cur_pri, [week]])),dtype='int32'))
        grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
        #grid.head()
        # Calculate number of hours spent
        self.dataset['business_duration'] = self.dataset['business_duration']//3600
        self.dataset_filt = self.dataset[['week','category','priority','business_duration']]
        self.dataset_filt.isna().sum()
        #grid.shape
        #dataset_filt.dtypes
        self.dataset_filt = self.dataset_filt.groupby(['week','category','priority']).agg({'business_duration': ['sum', 'count']})
        self.dataset_filt.columns = ['hours','count']
        self.dataset_filt = self.dataset_filt.reset_index()
        self.dataset_filt = pd.merge(grid,self.dataset_filt,on=['week','category','priority'],how='left').fillna(0)
        self.dataset_filt.shape
        #week47=dataset_filt[dataset_filt['category']==0]
        #week47[week47['week']==47]
        lag_variables  = ['hours','count']
        lags = [1 ,2 ,3 ,4 ,5 ,12]
        self.dataset_week = self.dataset_filt.copy()
        for lag in lags:
            self.dataset_week_new_df = self.dataset_filt.copy()
            self.dataset_week_new_df.hours+=lag
            self.dataset_week_new_df = self.dataset_week_new_df[['week','category','priority']+lag_variables]
            self.dataset_week_new_df.columns = ['week','category','priority']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
            self.dataset_week = pd.merge(self.dataset_week, self.dataset_week_new_df,on=['week','category','priority'] ,how='left')
        #dataset_week.shape
        self.dataset_week = self.dataset_week.fillna(0)
        self.X_train, self.X_cv, self.Y_train, self.Y_cv = self._create_train_and_test_set_for_incidents()
        
    def _create_train_and_test_set_for_incidents(self):    
        X_train = self.dataset_week[self.dataset_week['week']<47]
        X_cv =  self.dataset_week[self.dataset_week['week']>=47]
        Y_train = X_train['hours']
        Y_cv = X_cv['hours']
        del X_train['hours']
        del X_cv['hours']
        return X_train, X_cv, Y_train, Y_cv
    #@app.route('/getWeeklyReports/', methods=["GET"])    
    def get_weekly_hours(self):  
        self._preapre_incident_dataset()     
        model_dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_split=5,
                           max_leaf_nodes=10)
        param_grid = {"criterion": ["mse", "mae"],
              "min_samples_split": [10, 20, 40],
              "max_depth": [2, 6, 8],
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100],
              }
        model = GridSearchCV(model_dt, param_grid, cv=5)
        model.fit(self.X_train, self.Y_train)
        ypred = model.predict(self.X_cv)
        c_score = r2_score(self.Y_cv, ypred)
        #print(r2_score)
        weekly_hours=pd.DataFrame(data={"week":self.X_cv['week'],"category":self.X_cv['category'],"priority":self.X_cv['priority'],"Actual_hours":self.Y_cv,"Actual_Resources":self.Y_cv//8,"Predicted_hours":ypred,"Predicetd_Resources":ypred//8})
        weekly_hours['Predicted_hours']= weekly_hours['Predicted_hours'].astype(int)
        weekly_hours['Actual_hours']= weekly_hours['Actual_hours'].astype(int)
        weekly_hours['Actual_Resources']= weekly_hours['Actual_Resources'].astype(int)
        weekly_hours['Predicetd_Resources']= weekly_hours['Predicetd_Resources'].astype(int)
        weekly_hours['priority'] = weekly_hours['priority'].map(self.le_pri_mapping)
        weekly_hours['category'] = weekly_hours['category'].map(self.le_cat_mapping)
        #year_week = str(year) +'-'+ str(week) +'-'+ str(1)
        weekly_hours['week'] = str(self.year) +'-'+ weekly_hours['week'].astype(str) +'-'+ str(1)
        weekly_hours['week'] = pd.to_datetime(weekly_hours.week,format = '%Y-%W-%w')
        weekly_hours.to_csv("weekly_hours_spent_v1.csv",index=None)
        weekly_hours_dic = weekly_hours.to_dict('list')
        return weekly_hours
        #return jsonify(weekly_hours_dic)

if __name__ == '__main__':
    RS = ReportService()
    weekly_report = RS.get_weekly_hours()
    #app.run()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from flask import Flask, request, jsonify
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
from flask import Flask
app = Flask(__name__)
#dataset.shape
@app.route('/getWeeklyReports/', methods=["GET"])
def get_weekly_hours():
    dataset = pd.read_csv('C:\\Users\\Priya\\Desktop\\Sivisoft\\Time Model\\weekly_hours_spent.csv') 
    dataset = dataset.to_dict('list')
    return jsonify(dataset)
@app.route("/")
def hey_test():
    return "Working!"
if __name__ == "__main__":
    app.run()


# In[ ]:





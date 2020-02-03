import pandas as pd
from flask import Flask, request
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
from flask import Flask
app = Flask(__name__)
#dataset.shape
@app.route('/getWeeklyReports/', methods=["GET"])
def get_weekly_hours():
    #dataset = pd.read_csv('C:\\Users\\Priya\\Desktop\\Sivisoft\\Time Model\\weekly_hours_spent.csv')
    dataset = pd.read_csv('./weekly_hours_spent_v1.csv') 
    return dataset.to_html(header="true", table_id="table")
@app.route("/")
def hey_test():
    return "Working!"
if __name__ == "__main__":
    app.run(host='0.0.0.0')


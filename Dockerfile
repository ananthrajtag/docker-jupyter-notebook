FROM python:3

ADD Weekly_hours_prediction_v1.py /
ADD web_service_call.py /
ADD incident.xlsx /
RUN pip install pystrich pandas matplotlib seaborn sklearn sklearn flask flask_cors xlrd
RUN pip install more-itertools
RUN python Weekly_hours_prediction_v1.py  
EXPOSE 5000
CMD [ "python", "./web_service_call.py" ]

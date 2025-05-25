# it will read the data from notebook and upload it to mongoDB

from pymongo.mongo_client import MongoClient
import pandas as pd
import json

#url
# uri="mongodb+srv://imran:TdPLW9Ad0OzpSSD2@cluster0.fv0lm61.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

uri = "mongodb+srv://yashgupta9798:Yash9798@cluster0.7epp6hx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

#create a new client and connectt to server
client = MongoClient(uri)

#create database name and collection name
DATABASE_NAME="SensorDetection"
COLLECTION_NAME='waferfault'

df=pd.read_csv("D:\ML_Projects\sensorProject\notebooks\wafer_23012020_041211.csv")

df=df.drop("Unnamed: 0",axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
import json
import os

import pandas as pd
from pymongo import MongoClient


class CausalityModel:

    def __init__(self):
        try:
            conn = MongoClient()
            print("Connected successfully!!!")
        except:
            print("Could not connect to MongoDB")

        # database
        db = conn.database
        f = open(os.path.join(BA, 'filename'))
        f = open('../covid_data.json')
        data = json.load(f)
        collection = db.covid_cases
        countries = set()
        for i in data:
            new_record = {
                "_id":i['id'],
                "data":i['date'],
                "country":i['country'],
                "cases":i['cases'],
                "deaths":i['deaths'],
            }
            countries.add(i['country'])

        country_dataFrames = dict()
        for country in countries:
            country_data = collection.find({'country':country})
            country_dataFrames[country] = pd.DataFrame(country_data)

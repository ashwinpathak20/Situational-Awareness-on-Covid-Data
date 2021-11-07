from pymongo import MongoClient
import json
import os
import matplotlib.pyplot as plt

try:
    conn = MongoClient()
    print("Connected successfully!!!")
except:  
    print("Could not connect to MongoDB")
  
# database
db = conn.database
f = open('us-states.json')
data = json.load(f)
collection = db.covid_cases
states = set()


for i in data:
    new_record = {
        "_id":i['id'],
        "data":i['Date'],
        "state":i['state'],
        "fips":i['fips'],
        "cases":i['cases'],
        "deaths":i['deaths'],
    }
    states.add(i['state'])
    #rec =  collection.insert_one(new_record)
    #print("Data inserted with record ids",rec)
  
# Printing the data inserted
#cursor = collection.find()
for state1 in states:
    print(state1)
    state_data = collection.find({'state':state1})
    x = []
    y1 = []
    y2 = []
    for i in state_data:
        x.append(i['data'])
        y1.append(int(i['cases']))
        y2.append(int(i['deaths']))
    for i in range(len(y1)-1,1,-1):
        y1[i] = max(0,y1[i]-y1[i-1])
        y2[i] = max(0,y2[i]-y2[i-1])
    plt.plot(x, y1)
    plt.xticks([])
    plt.xlabel('date')
    plt.ylabel('cases')
    plt.title('cases for ' + state1)
    path = 'graphs_time/'+state1
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    plt.savefig('graphs_time/'+state1+'/cases.png')
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(x, y2)
    plt.xticks([])
    plt.xlabel('date')
    plt.ylabel('deaths')
    plt.title('deaths for ' + state1)
    plt.savefig('graphs_time/'+state1+'/deaths.png')
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y1 = []
    y2 = []

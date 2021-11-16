import csv
import json
import time

# TODO: Add mechanism to resume fetching from the last fetched date
def csvToJson(csv_path, json_path): # date_file_path, last_fetched_date):

    us_data = []
    # max_date_fetched = last_fetched_date
    countries = ["India", "China", "United States", "European Union"]
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for rows in csv_reader:
            # curr_date = time.strptime(rows["Date"], "%m/%d/%y")
            # max_date_fetched = max(curr_date, max_date_fetched)
            if rows["country"] in countries: #curr_date > last_fetched_date and
                rows["id"] = rows["date"] + "#" + rows["country"]
                if rows["deaths"] == "":
                    rows["deaths"] = 0
                us_data.append(rows)

    with open(json_path, 'a', encoding='utf-8') as json_file:
        json_file.write(json.dumps(us_data, indent=4))

    # with open(date_file_path, 'w', encoding='utf-8') as date_file:
    #     date_file.write(str(max_date_fetched))


csv_path = r'covid_data.csv'
json_path = r'covid_data.json'
# date_file_path = r'date.txt'

# last_fetched_date = time.strptime("11/4/21", "%m/%d/%y")
csvToJson(csv_path, json_path)
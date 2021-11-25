import datetime
import json
import pandas as pd

COVID_CSV_DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"


class CovidDataAutomation:

    def __init__(self, url):
        self.url = url
        self.required_columns = ['location', 'date', 'total_cases_per_million', 'total_deaths_per_million']
        self.allowed_countries = ["India", "China", "United States", "European Union"]

    def run(self):
        print("========================================")
        print("Loading Data...")
        print("----------------------------------------")
        self.load_data()
        print("Starting to process data...")
        self.process_incrementally()
        print("----------------------------------------")
        print("Writing output to json file...")
        self.write_json_file()
        print("========================================")
        print("Processed {} Covid Data Records successfully".format(self.total_dataset_size))
        print("")
        print("Execution Completed.")

    @staticmethod
    def get_start_of_century():
        return '2000-01-01'

    @staticmethod
    def read_from_local_covid_data_file():
        with open('last_fetched_date.txt', 'r') as file:
            while (line := file.readline().rstrip()):
                yield line
                return

    @staticmethod
    def get_last_date_fetched():
        last_fetched_date = "2000-01-01"
        try:
            last_fetched_date = [x for x in CovidDataAutomation.read_from_local_covid_data_file()][0]
            if not last_fetched_date or len(last_fetched_date.strip()) == 0:
                last_fetched_date = CovidDataAutomation.get_start_of_century()

        finally:
            print("----------------------------------------")
            print("Processing records after {}".format(last_fetched_date))
            return last_fetched_date

    @staticmethod
    def store_last_date_fetched(last_fetched_date):
        with open("last_fetched_date.txt", "w") as file:
            file.write(last_fetched_date)

    @staticmethod
    def get_column_name_mappings():
        return {
            'location': 'country',
            'total_cases_per_million': 'total_cases',
            'total_deaths_per_million': 'total_deaths'
        }

    def load_data(self):
        self.data = pd.read_csv(self.url)

    def write_json_file(self):
        records = self.data.to_json(orient='records')
        records = json.loads(records)

        with open('covid_data.json', 'a') as file:
            for record in records:
                file.write(json.dumps(record))
                file.write("\n")

    def process_incrementally(self):
        self.data = self.data[self.required_columns] \
            .rename(columns=CovidDataAutomation.get_column_name_mappings()) \
            .fillna(0)
        self.data = self.data[(self.data["country"].isin(self.allowed_countries))] \
            .sort_values(by='date', ascending=False)

        last_fetched_date = CovidDataAutomation.get_last_date_fetched()
        CovidDataAutomation.store_last_date_fetched(self.data["date"].values[0])

        print("----------------------------------------")
        print("Fetching the last fetched date...")
        self.data = self.data[self.data['date'] > last_fetched_date]
        self.data['id'] = self.data['country'] + "#" + self.data['date']
        self.total_dataset_size = len(self.data.index)



def main():
    automation_agent = CovidDataAutomation(COVID_CSV_DATA_URL)
    automation_agent.run()


if __name__ == '__main__':
    main()

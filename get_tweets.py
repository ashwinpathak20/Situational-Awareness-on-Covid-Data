import requests
import os
import json
import pandas as pd
import numpy as np
from csv import writer

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = "AAAAAAAAAAAAAAAAAAAAAJUCUQEAAAAAKbKDbiYVdLKbeg0czFTSDHM02cQ%3DubcnsoHHQ9YWszx5bZslGmhktsHlxa0rbZO26shLQs2jtZRcjV"


def create_url(x_str):
    tweet_fields = "tweet.fields=lang,author_id"
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    ids = "ids="+x_str
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def main():
    df = pd.read_csv('/Users/ashwinpathak/Downloads/corona_tweets_01.csv')
    a = df.to_numpy()
    l = len(a)
    p1 = 0
    p2 = 100
    while p2 < l :
        x_arrstr = np.char.mod('%d', a[p1:p2,0])
        x_str = ",".join(x_arrstr)
        url = create_url(x_str)
        json_response = connect_to_endpoint(url)
        for j in json_response['data']:
            t = a[p1:p2,0]
            idx = p1+np.argwhere(t==int(j["id"]))[0][0]
            text = j["text"]
            val = a[idx,1]
            row = []
            row.append(j["id"])
            row.append(text)
            row.append(str(val))
            with open('document2.csv', 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(row)
        p1+=100
        p2+=100
    

if __name__ == "__main__":
    main()

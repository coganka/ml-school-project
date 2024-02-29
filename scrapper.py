import requests
import time
from datetime import datetime as dt

def get_data():
    #get input
    company = input('Enter company stock name: ')
    from_date = input('Enter start date in yy/mm/dd format: ')
    to_date = input('Enter end date in yy/mm/dd format: ')

    #datetime objs
    from_dt = dt.strptime(from_date, '%Y/%m/%d')
    to_dt = dt.strptime(to_date, '%Y/%m/%d')
    #Unix timestamp format
    from_epoch = int(time.mktime(from_dt.timetuple()))
    to_epoch = int(time.mktime(to_dt.timetuple()))

    url = f"https://query1.finance.yahoo.com/v7/finance/download/{company.upper()}?period1={from_epoch}&period2={to_epoch}&interval=1d&events=history&includeAdjustedClose=true"

    #required headers
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"}

    content = requests.get(url, headers=headers).content
    print(content)

    with open('data.csv', 'wb') as file:
        file.write(content)
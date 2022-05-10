import pandas as pd
import math
import time
import os
from tqdm import tqdm
from bitmex import bitmex
from datetime import timedelta
from dateutil import parser


class BitMexExporter:
    """BitMex Exporter"""
    def __init__(self):
        self.client = bitmex(test=False,
                             api_key=os.environ.get('BITMEX_API_KEY'),
                             api_secret=os.environ.get('BITMEX_API_SECRET'))
        self.binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
        self.batch_size = 750

    def minutes_of_new_data(self, symbol, kline_size, data):
        if len(data) > 0:
            old = parser.parse(data["timestamp"].iloc[-1])
        old = self.client.Trade.Trade_getBucketed(
            symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
        new = self.client.Trade.Trade_getBucketed(
            symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
        return old, new

    def export(self, symbol, kline_size, save=False):
        filename = '%s-%s-data.csv' % (symbol, kline_size)
        if os.path.isfile(filename):
            data_df = pd.read_csv(filename)
        else:
            data_df = pd.DataFrame()
        oldest_point, newest_point = self.minutes_of_new_data(symbol, kline_size, data_df)
        delta_min = (newest_point - oldest_point).total_seconds() / 60
        available_data = math.ceil(delta_min / self.binsizes[kline_size])
        rounds = math.ceil(available_data / self.batch_size)
        if rounds > 0:
            print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data in %d rounds.' % (
                delta_min, symbol, available_data, kline_size, rounds))
            for round_num in tqdm(range(rounds)):
                time.sleep(1)
                new_time = (oldest_point + timedelta(minutes=round_num * self.batch_size * self.binsizes[kline_size]))
                data = self.client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size,
                                                           count=self.batch_size, startTime=new_time).result()[0]
                temp_df = pd.DataFrame(data)
                data_df = data_df.append(temp_df)
        data_df.set_index('timestamp', inplace=True)
        if save and rounds > 0:
            data_df.to_csv(filename)
        return data_df

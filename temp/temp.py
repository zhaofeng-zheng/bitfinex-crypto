import pandas as pd
import numpy as np
import time


df = pd.read_csv('temp.csv', index_col=[0], parse_dates=['candle_begin_time'])
for i in range(100):
    start = time.perf_counter()
    # df[['candle_begin_time', 'equity_curve']].to_hdf('temp.h5', key='equity_curve' + str(i))
    df[['candle_begin_time', 'equity_curve']].to_csv('equity_curve' + str(i) + '.csv.gz', index=False, compression='gzip')
    end = time.perf_counter()
    print(f'{end - start}')

print(pd.HDFStore('temp.h5', mode='r').keys())



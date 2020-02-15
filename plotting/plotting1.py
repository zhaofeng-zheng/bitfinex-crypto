import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from pathlib import Path
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(figsize=(20, 15))
for equity_curve in Path('../data/all params equity curve').iterdir():
    print('正在绘制', str(equity_curve))
    df = pd.read_csv(equity_curve, parse_dates=['candle_begin_time'])
    plt.plot(df['candle_begin_time'], df['equity_curve'])
    plt.show()
    time.sleep(1)
    plt.close()



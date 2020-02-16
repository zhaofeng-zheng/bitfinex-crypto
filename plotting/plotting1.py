import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

h5 = pd.HDFStore('../data/all params equity curve/all_params_equity_curve.h5', mode='r')
print(len(h5.keys()))
for equity_curve in h5.keys():
    print('正在绘制', str(equity_curve))
    print(h5[equity_curve])
h5.close()


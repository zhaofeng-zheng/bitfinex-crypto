import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1000)


def cal_common_backtest_params(df):
    '''
    输入有equity_curve, signal的df, 输出回测参数, 例如最大亏损率,连续亏损率
    :param df:
    :return:
    '''
    t = df.loc[df['signal'].notnull(), ['signal', 'equity_curve']]
    t['signal'] = t['signal'].astype('int64')
    t = t[t['signal'] != 0]

    # calculate the percent change in equity after each long_short operation,
    # the figure on the 'everytime' column corresponds to each previous market
    # operation.
    t['everytime_kai_ping_equity_change'] = t['equity_curve'].pct_change(1)
    t.fillna(value=0, inplace=True)

    # remove the ones where there is no profit nor loss
    t = t[t['everytime_kai_ping_equity_change'] != 0]
    t.loc[t['everytime_kai_ping_equity_change'] > 0, 'PoN'] = 1
    t.loc[t['everytime_kai_ping_equity_change'] < 0, 'PoN'] = -1

    # t includes a subgroup column which labels all the
    # consecutive profits and losses using an integer
    # t is the MOST IMPORTANT DATAFRAME THERE IS
    t['subgroup'] = (t['PoN'] != t['PoN'].shift(1)).cumsum()

    # t2 is essentially t without profits, so we only focus on the losses here
    t2 = t[t['PoN'] < 0]
    # print(t2)

    # print(t2.groupby('subgroup', as_index=False)['subgroup'].size())

    max_consecutive_loss = t2.groupby('subgroup', as_index=False)['subgroup'].size().max()
    profit_number = t[t['everytime_kai_ping_equity_change'] > 0]['signal'].count()
    loss_number = t[t['everytime_kai_ping_equity_change'] < 0]['signal'].count()
    max_profit_rate = t['everytime_kai_ping_equity_change'].max()
    max_loss_rate = t['everytime_kai_ping_equity_change'].min()

    expected_payoff = (t['everytime_kai_ping_equity_change'] / len(t)).sum()

    common_backtest_param = [max_consecutive_loss, profit_number, loss_number, max_profit_rate, max_loss_rate, expected_payoff]
    return common_backtest_param


def cal_advc_backtest_params(df, ATR_period = 100):
    '''
    Takes in a standard df and splits out parameters in the form of a list
    :param df:
    :param ATR_period:
    :return:
    '''
    d = df.copy()
    d['previous_close'] = d['close'].shift().fillna(d.iloc[0]['close'])

    d['high-low'] = d['high'] - d['low']
    d['high-previous_close'] = d['high'] - d['previous_close']
    d['low-previous_close'] = d['low'] - d['previous_close']
    d['TR'] = d[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
    d['ATR'] = d['TR'].rolling(ATR_period, min_periods=1).mean()

    d.drop(['TR', 'high-low', 'previous_close', 'high-previous_close', 'low-previous_close'], axis=1, inplace=True)

    d.loc[(d['pos'] != d['pos'].shift()) & (df['pos'] != 0), 'start_time'] = df['candle_begin_time']

    E_ratio_lst = []
    for MFE_MAE_period in [288, 576, 1440, 2016]:
        tmp = d.copy()
        tmp['start_time'].fillna(method='ffill', inplace=True, limit=MFE_MAE_period-1)
        tmp.loc[tmp['pos'] == 0, 'start_time'] = np.NaN
        t = tmp.groupby('start_time').apply(lambda x: x['close'] - x.iloc[0]['open'])
        t = t.reset_index(level=[0])

        tmp['price_change_after_signal'] = t['close']
        tmp = tmp.astype({'pos': 'int64'})
        t = pd.DataFrame()
        for i, j in tmp.groupby('start_time'):
            j['MFE'] = np.NaN
            j['MAE'] = np.NaN
            if j.iloc[0]['pos'] == -1:
                j.iloc[0, -2] = abs(min(j['price_change_after_signal'].min(), 0))
                j.iloc[0, -1] = abs(max(j['price_change_after_signal'].max(), 0))
            else:
                j.iloc[0, -2] = abs(max(j['price_change_after_signal'].max(), 0))
                j.iloc[0, -1] = abs(min(j['price_change_after_signal'].min(), 0))
            t = t.append(j)
        tmp[['MFE', 'MAE']] = t[['MFE', 'MAE']]
        tmp['adjusted_MFE'] = tmp['MFE'] / tmp['ATR']
        tmp['adjusted_MAE'] = tmp['MAE'] / tmp['ATR']
        average_adjusted_MFE = tmp['adjusted_MFE'].sum() / len(tmp['adjusted_MFE'].dropna())
        average_adjusted_MAE = tmp['adjusted_MAE'].sum() / len(tmp['adjusted_MAE'].dropna())
        E_ratio = average_adjusted_MFE / average_adjusted_MAE
        print(E_ratio)
        E_ratio_lst.append(E_ratio)
    return E_ratio_lst
#
# df = pd.read_csv(r'C:\Users\19750\Desktop\test\crypto\program\bfx_eosusd_bolling_parameter_screen_project\temporary.csv')
# cal_common_backtest_params(df)


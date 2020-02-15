import pandas as pd
pd.set_option('expand_frame_repr', False)
# pd.set_option('display.max_row', 1000)

def cal_max_drawdown(df):
    '''
    输入含有equity_curve列的df, 返回最大回撤开始和结束日期
    :param df:
    :return:
    '''
    df = df.copy()
    df['max_to_date'] = df['equity_curve'].expanding(min_periods=1).max()
    max_periods_between_highs = df.groupby(df['max_to_date']).size().max()
    df['max_drawdown_to_date'] = 1 - df['equity_curve'] / df['max_to_date']
    end_date, max_drawdown = tuple(df.sort_values(by='max_drawdown_to_date', ascending=False).iloc[0][['candle_begin_time', 'max_drawdown_to_date']])
    start_date = df[df['candle_begin_time'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['candle_begin_time']
    return max_periods_between_highs, max_drawdown, start_date, end_date

# df = pd.read_csv(r'C:\Users\19750\Desktop\test\crypto\program\bfx_eosusd_bolling_parameter_screen_project\temporary.csv')
# cal_max_drawdown(df)
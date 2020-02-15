import pandas as pd
import time
from ccxt import bitfinex
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing.pool import Pool
from Signal import Signal
from func.backtest_params import cal_common_backtest_params
from func.calculate_max_drawdown import cal_max_drawdown
from func.other_func import *


class Optimizer(bitfinex):
    def __init__(self, symbol, recentPerformanceTimeDuration, leverage_rate=3.3):
        super().__init__()
        self.symbol = symbol
        self.leverage_rate = leverage_rate
        self.recentPerformanceTimeDuration = recentPerformanceTimeDuration
        self.base_coin = symbol.split('/')[-1]
        self.trade_coin = symbol.split('/')[0]

    def fetch_all_candle(exchange, output_as_file: bool=False, timeframe='1m', since=0, limit=5000):
        '''
        用来爬取从最开始到现在的所有K线数据(可指定周期), 以DF形式返回 , 交易所回应会有延迟 ,太近的数据抓取不到,所以最后一行跟现在可能有十分钟左右的延迟,
        :param exchange:输入一个ccxt exchange对象
        :param symbol:
        :param since:
        :param limit:
        :return: df
        '''
        symbol = exchange.symbol
        # exchange.enableRateLimit = True
        df = pd.DataFrame()
        while True:
            start = time.perf_counter()
            _temp = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)  # 从该品种在该交易所最早K线开始抓取
            if _temp:  # 判断是否抓取到
                timestamp = _temp[-1][0]  # 该批最新K线的时间戳
                print('抓取到的最新K线是 : ', exchange.iso8601(timestamp))
                df = df.append(_temp, ignore_index=True)
            else:
                continue
            now_timestamp = time.time() * 1000  # 现在的时间戳
            if abs(timestamp - now_timestamp) > 60000 and _temp:  # 当爬取到信息, 且 , 该批K线最新时间戳与现在的时间相差大于1分钟时
                since = timestamp + 60000  # 从这批K线的最近下一根K线开始爬取
                print('程序会在以下数字小于60000时停止运行: \n', abs(timestamp - now_timestamp))
                print('进入下一轮\n')
                end = time.perf_counter()
                while end - start < 1:
                    end = time.perf_counter()
            else:
                # ==处理数据
                df = df.rename(columns={0: 'candle_begin_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
                df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'], unit='ms')
                df['candle_begin_time'] = df['candle_begin_time'] + timedelta(hours=11)
                df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
                if output_as_file:
                    df.to_csv(Path().parent / 'data/all_candles.csv')
                    return df
                else:
                    return df

    @staticmethod
    def generate_equity_curve(df, leverage_rate, c_rate, min_margin_rate=0.15):
        """
            可做多可做空情况下的资金曲线,例如bitfinex
            :param df:  带有signal和pos的原始数据
            :param leverage_rate:  bfx交易所最多提供3倍杠杆，leverage_rate可以在(0, 3]区间选择
            :param c_rate:  手续费
            :param min_margin_rate:  低保证金比例，必须占到借来资产的15%
            :return: df
        """
        # =====基本参数
        init_cash = 100  # 初始资金
        min_margin = init_cash * leverage_rate * min_margin_rate  # 最低保证金

        # =====根据pos计算资金曲线
        # ===计算涨跌幅
        df['change'] = df['close'].pct_change(1)  # 根据收盘价计算涨跌幅
        df['buy_at_open_change'] = df['close'] / df['open'] - 1  # 从今天开盘买入，到今天收盘的涨跌幅
        df['sell_next_open_change'] = df['open'].shift(-1) / df['close'] - 1  # 从今天收盘到明天开盘的涨跌幅
        df.at[len(df) - 1, 'sell_next_open_change'] = 0

        # ===选取开仓、平仓条件
        condition1 = df['pos'] != 0
        condition2 = df['pos'] != df['pos'].shift(1)
        open_pos_condition = condition1 & condition2

        condition1 = df['pos'] != 0
        condition2 = df['pos'] != df['pos'].shift(-1)
        close_pos_condition = condition1 & condition2

        # ===对每次交易进行分组
        df.loc[open_pos_condition, 'start_time'] = df['candle_begin_time']
        df['start_time'].fillna(method='ffill', inplace=True)
        df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

        # ===计算仓位变动
        # 开仓时仓位
        df.loc[open_pos_condition, 'position'] = init_cash * leverage_rate * (1 + df['buy_at_open_change'])  # 建仓后的仓位

        # 开仓后每天的仓位的变动
        group_num = len(df.groupby('start_time'))
        if group_num > 1:
            t = df.groupby('start_time').apply(lambda x: x['close'] / x.iloc[0]['close'] * x.iloc[0]['position'])
            t = t.reset_index(level=[0])
            df['position'] = t['close']
        elif group_num == 1:
            t = df.groupby('start_time')[['close', 'position']].apply(
                lambda x: x['close'] / x.iloc[0]['close'] * x.iloc[0]['position'])
            df['position'] = t.T.iloc[:, 0]

        # 每根K线仓位的最大值和最小值，针对最高价和最低价
        df['position_max'] = df['position'] * df['high'] / df['close']
        df['position_min'] = df['position'] * df['low'] / df['close']

        # 平仓时仓位
        df.loc[close_pos_condition, 'position'] *= (1 + df.loc[close_pos_condition, 'sell_next_open_change'])

        # ===计算每天实际持有资金的变化
        # 计算持仓利润
        df['profit'] = (df['position'] - init_cash * leverage_rate) * df['pos']  # 持仓盈利或者损失

        # 计算持仓利润最小值
        df.loc[df['pos'] == 1, 'profit_min'] = (df['position_min'] - init_cash * leverage_rate) * df[
            'pos']  # 最小持仓盈利或者损失
        df.loc[df['pos'] == -1, 'profit_min'] = (df['position_max'] - init_cash * leverage_rate) * df[
            'pos']  # 最小持仓盈利或者损失

        # 计算实际资金量
        df['cash'] = init_cash + df['profit']  # 实际资金
        df['cash'] -= init_cash * leverage_rate * c_rate  # 减去建仓时的手续费
        df['cash_min'] = df['cash'] - (df['profit'] - df['profit_min'])  # 实际最小资金
        df.loc[close_pos_condition, 'cash'] -= df.loc[close_pos_condition, 'position'] * c_rate  # 减去平仓时的手续费

        # ===判断是否会爆仓
        _index = df[df['cash_min'] <= min_margin].index
        if len(_index) > 0:
            print('有爆仓')
            df.loc[_index, '强平'] = 1
            df['强平'] = df.groupby('start_time')['强平'].fillna(method='ffill')
            df.loc[(df['强平'] == 1) & (df['强平'].shift(1) != 1), 'cash_强平'] = df['cash_min']  # 此处是有问题的
            df.loc[(df['pos'] != 0) & (df['强平'] == 1), 'cash'] = None
            df['cash'].fillna(value=df['cash_强平'], inplace=True)
            df['cash'] = df.groupby('start_time')['cash'].fillna(method='ffill')
            df.drop(['强平', 'cash_强平'], axis=1, inplace=True)  # 删除不必要的数据

        # ===计算资金曲线
        df['equity_change'] = df['cash'].pct_change()
        df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'cash'] / init_cash - 1  # 开仓日的收益率
        df['equity_change'].fillna(value=0, inplace=True)
        df['equity_curve'] = (1 + df['equity_change']).cumprod()

        # ===删除不必要的数据
        df.drop(['change', 'buy_at_open_change', 'sell_next_open_change', 'start_time', 'position', 'position_max',
                 'position_min', 'profit', 'profit_min', 'cash', 'cash_min'], axis=1, inplace=True)

        return df

    def _time_interval_based_multiprocessing(self, r: int, params_lst: list, now: datetime = datetime.now()):
        '''
        :param r:
        :return: Void
        '''
        # 用的信号
        # ///
        signal_fn = Signal.modified_bolling_signal_bt
        # ///
        leverage_rate = self.leverage_rate
        rtn = pd.DataFrame()
        if r == 60:
            rule_type = "1H"
        else:
            rule_type = str(r) + 'T'
        df = pd.read_csv(r'./data/all_candles.csv', parse_dates=['candle_begin_time'])
        df = df[df['candle_begin_time'] >= pd.to_datetime('2017/01/01')]
        d = self.convert_date_period(df, rule_type=rule_type)
        for params in params_lst:
            params = list(map(lambda x: round(x, 3), params))
            n = params[0]
            m = params[1]
            df = d.copy()
            print('当前时间周期是 : ', rule_type)
            print(params)
            df = signal_fn(df, params)
            df = self.generate_equity_curve(df, leverage_rate=leverage_rate, c_rate=2 / 1000)
            # ====最大收益和最终受益(以及其他策略参数), 并打印出来
            PoL = df.iloc[-1]['equity_curve']
            max_profit = df['equity_curve'].max()
            kongcang_pct = df['pos'].value_counts()[-1] / len(df)
            chicang_pct = 1 - kongcang_pct
            # ==计算最近指定时间的资金曲线变化百分比
            # day_slice作用， 检查过去10天到过去5天的策略表现时，相差5天（10 - 5 = 5）， 就用day_slice表示
            day_slice = 5
            for i, days in enumerate(range(5, self.recentPerformanceTimeDuration + 1, day_slice)):
                TruncatedDfForCalcEquityChange: pd.DataFrame = df[
                    (df['candle_begin_time'] >= now - timedelta(days=days)) &
                    (df['candle_begin_time'] < now - timedelta(days=days - day_slice))]
                startEquity: float = TruncatedDfForCalcEquityChange.iat[0, -1]
                endEquity: float = TruncatedDfForCalcEquityChange.iat[-1, -1]
                equityPctChange: float = round((endEquity - startEquity) / startEquity, 3)
                rtn.loc[str(params), f'past {days} to {days - day_slice } days equity change'] \
                    = equityPctChange
                # ==计算最近指定时间开仓总次数
                # ///
                # TruncatedDfForCalcEquityChangeWithSignalClmnFfilled = TruncatedDfForCalcEquityChange['signal'].fillna(method='ffill')
                # FfilledSignal: pd.Series = TruncatedDfForCalcEquityChangeWithSignalClmnFfilled
                # FfilledSignal = FfilledSignal.replace(to_replace=0, value=np.NaN)
                # ((FfilledSignal != FfilledSignal.shift(1)) & ~FfilledSignal.isna()).value_counts()
                # 由于信号生成函数signal bolling stop loss的特殊性，导致signal不需要去重，以上代码没有必要， 但还是有价值的。
                # ///
                longPositionTimes = TruncatedDfForCalcEquityChange['signal'].value_counts().get(1.0, default=0)
                shortPositionTimes = TruncatedDfForCalcEquityChange['signal'].value_counts().get(-1.0, default=0)
                timesOpenPosition = longPositionTimes + shortPositionTimes
                rtn.loc[str(params), f'past {days} to {days - day_slice} days open position times'] \
                    = timesOpenPosition
                # ==计算最近指定时间内最大回撤
                if i == len(range(5, self.recentPerformanceTimeDuration + 1, 5)) - 1:
                    drawdownInSpecifiedTime: tuple = cal_max_drawdown(TruncatedDfForCalcEquityChange)
                    rtn.loc[str(params), f'past {self.recentPerformanceTimeDuration} days max periods between highs'] \
                        = drawdownInSpecifiedTime[0]
                    rtn.loc[str(params), f'past {self.recentPerformanceTimeDuration} days max drawdown'] \
                        = drawdownInSpecifiedTime[1]
                    rtn.loc[str(params), f'past {self.recentPerformanceTimeDuration} days max drawdown start date'] \
                        = drawdownInSpecifiedTime[2]
                    rtn.loc[str(params), f'past {self.recentPerformanceTimeDuration} days max drawdown end date'] \
                        = drawdownInSpecifiedTime[3]
            # ====常见策略回测参数计算, 使用common backtesting params 这个函数
            common_backtest_params = cal_common_backtest_params(df=df)
            max_consecutive_loss = common_backtest_params[0]  # 最大连续亏损周期
            profit_number = common_backtest_params[1]  # 盈利次数
            loss_number = common_backtest_params[2]  # 亏损次数
            max_profit_rate = common_backtest_params[3]  # 单笔最大盈利
            max_loss_rate = common_backtest_params[4]  # 单笔最大亏损
            expected_payoff = common_backtest_params[5]  # 持仓期望盈利/亏损率
            profit_freq = profit_number / (profit_number + loss_number)  # 胜率
            loss_freq = loss_number / (profit_number + loss_number)  # 赔率
            # ==计算最大回撤
            max_drawdown_lst = cal_max_drawdown(df)
            max_periods_between_highs = max_drawdown_lst[0]
            max_drawdown = max_drawdown_lst[1]
            start_date = max_drawdown_lst[2]
            end_date = max_drawdown_lst[3]
            # 将策略参数写进DataFrame里面
            rtn.loc[str(params), 'n'] = n
            rtn.loc[str(params), 'm'] = m
            rtn.loc[str(params), 'PoL'] = PoL
            rtn.loc[str(params), 'max_profit'] = max_profit
            rtn.loc[str(params), 'kongcang_pct'] = kongcang_pct
            rtn.loc[str(params), 'chicang_pct'] = chicang_pct
            rtn.loc[str(params), 'max_consecutive_loss'] = max_consecutive_loss
            rtn.loc[str(params), 'profit_number'] = profit_number
            rtn.loc[str(params), 'loss_number'] = loss_number
            rtn.loc[str(params), 'max_profit_rate'] = max_profit_rate
            rtn.loc[str(params), 'max_loss_rate'] = max_loss_rate
            rtn.loc[str(params), 'expected_payoff'] = expected_payoff
            rtn.loc[str(params), 'max_periods_between_highs'] = max_periods_between_highs
            rtn.loc[str(params), 'profit_freq'] = profit_freq
            rtn.loc[str(params), 'loss_freq'] = loss_freq
            rtn.loc[str(params), 'max_drawdown'] = max_drawdown
            rtn.loc[str(params), 'start_date'] = start_date
            rtn.loc[str(params), 'end_date'] = end_date
            df[['candle_begin_time', 'equity_curve']].to_hdf(f'./data/all params equity curve/all_equity_curve.h5', key=f'{self.trade_coin}_{rule_type}_{str(params)}_equity_curve', mode='a')
        if os.path.exists(r'./data/backtest results/parameter_result_' + rule_type + '.csv'):
            rtn.to_csv(r'./data/backtest results/parameter_result_' + rule_type + '.csv', header=False, mode='a')
        else:
            rtn.to_csv(r'./data/backtest results/parameter_result_' + rule_type + '.csv', header=True, mode='a')

    def screen_for_best_parameter(self):
        '''
        Output backtest result as file
        :return: None
        '''
        now = datetime.now()
        r_lst = [5, 15, 30, 60]
        all_params = [[(n, m * 0.1, pingcang_multiplier * 0.1)
                       for n in range(step, step + 500, 10)
                       for m in range(15, 31, 1)
                       for pingcang_multiplier in range(5, 16, 1) if m > pingcang_multiplier] for step in range(500, 2501, 500)]
        for r in r_lst:
            if r == 60:
                rule_type = "1H"
            else:
                rule_type = str(r) + 'T'
            if os.path.exists(r'./data/backtest results/parameter_result_' + rule_type + '.csv'):
                shutil.move(r'./data/backtest results/parameter_result_' + rule_type + '.csv',
                            r'./data/temp/parameter_result_' + rule_type + '.csv')
            pool = Pool(processes=os.cpu_count()-1)
            for params_lst in all_params:
                pool.apply_async(func=self._time_interval_based_multiprocessing, args=(r, params_lst), kwds={'now': now}, error_callback=self.error_callback)
            pool.close()
            pool.join()

    @staticmethod
    def convert_date_period(df, rule_type, base=0):
        '''
        将原始数据的数据周期转为rule_type所指定的数据周期, 并返回处理好的df
        :param df:
        :param rule_type:
        :param base:
        :return:
        '''

        period_df = df.resample(rule=rule_type, on='candle_begin_time', base=base, label='left', closed='left').agg(
            {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        )
        period_df.reset_index(inplace=True)
        period_df.dropna(subset=['open', 'high', 'low', 'close'], how='any', inplace=True)
        period_df = period_df[period_df['volume'] > 0]
        period_df.reset_index(drop=True, inplace=True)
        df = period_df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]

        # 选定特定时间
        # df = df.loc[df['candle_begin_time'] >= pd.to_datetime('20170101')]
        # df.reset_index(inplace=True, drop=True)

        return df

    def error_callback(self, error):
        raise error



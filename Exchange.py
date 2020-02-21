from ccxt import bitfinex
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from time import sleep
import csv
from pathlib import Path
import warnings
from multiprocessing.pool import ThreadPool
import os

from Signal import Signal

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)


class MyExchange(bitfinex):
    def __init__(self, symbol, timeInterval, params=(2030, 2, 9), leverage_rate=3.3,
                 apiKey='6LjLCpcybWA5If7MKSOmVEwRfXmXYqeJKg1yvaAJf88',
                 secret='GCHXtdepaJStLWI7AsCrIXNvVSywCIfZlAGQEheimPW'):
        super().__init__()
        self.apiKey = apiKey
        self.secret = secret
        self.symbol = symbol
        self.leverage_rate = leverage_rate
        self.timeInterval = timeInterval
        self.base_coin = symbol.split('/')[-1]
        self.trade_coin = symbol.split('/')[0]
        self.params = params

    @staticmethod
    def next_run_time(time_interval, initial_min=0, ahead_time=0):
        time_interval = '60m' if time_interval == '1h' else time_interval
        if time_interval.endswith('m'):
            now_time = datetime.now()
            time_interval = int(time_interval.strip('m'))
            if initial_min < time_interval:
                if now_time.minute % time_interval >= initial_min:
                    target_minute = initial_min + (int(now_time.minute / time_interval) + 1) * time_interval
                else:
                    target_minute = now_time.minute - now_time.minute % time_interval + initial_min
            else:
                raise ValueError('initial_min必须小于time_interval')

            if target_minute < 60:
                target_time = now_time.replace(minute=target_minute, second=0, microsecond=0)
            else:
                if now_time.hour == 23:
                    target_time = now_time.replace(hour=0, minute=initial_min, second=0, microsecond=0)
                    target_time = target_time + timedelta(days=1)
                else:
                    target_time = now_time.replace(hour=now_time.hour + 1, minute=initial_min, second=0, microsecond=0)
            print('下次运行时间 : %s' % target_time)
            return target_time
        else:
            print("time_interval应以m结尾")

    def fetch_my_balance(self):
        while True:
            # 抓取普通交易账户余额
            exchange_balance = self.fetch_balance({'type': 'exchange'})['total']
            try:
                trade_coin_balance = exchange_balance[self.trade_coin]
            except KeyError:
                trade_coin_balance = 0
            try:
                base_coin_balance = exchange_balance[self.base_coin]
            except KeyError:
                base_coin_balance = 0
            print(self.base_coin, '在bitfinex普通交易账户余额为:', base_coin_balance)
            print(self.trade_coin, '在bitfinex普通交易账户余额为:', trade_coin_balance)

            # 抓取保证金交易账户余额
            margin_balance = self.fetch_balance({'type': 'trading'})['total']
            try:
                trade_coin_margin_balance = margin_balance[self.trade_coin]
            except KeyError:
                trade_coin_margin_balance = 0
            try:
                base_coin_margin_balance = margin_balance[self.base_coin]
            except KeyError:
                base_coin_margin_balance = 0
            print(self.base_coin, '在bitfinex保证金交易账户余额为:', base_coin_margin_balance)
            print(self.trade_coin, '在bitfinex保证金交易账户余额为:', trade_coin_margin_balance)
            return trade_coin_balance, base_coin_balance, trade_coin_margin_balance, base_coin_margin_balance


    def parsed_bitfinex_recent_candles(self):
        while True:
            try:
                start = time.perf_counter()
                timeInterval_in_minutes = int(self.timeInterval.strip('m')) if self.timeInterval != '1h' else 60
                number_of_candles_needed = self.params[0] + 1000
                candle = self.fetch_ohlcv(symbol=self.symbol, timeframe=self.timeInterval,
                                          since=(time.time() - timeInterval_in_minutes * 60 * number_of_candles_needed) * 1000,
                                          limit=number_of_candles_needed)
                end = time.perf_counter()
                print(f'Fetched in {end - start} seconds')
                df = pd.DataFrame(candle)
                df['candle_begin_time'] = pd.to_datetime(df[0], unit='ms')
                df.rename(columns={1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
                df['candle_begin_time'] = df['candle_begin_time'] + timedelta(hours=11)
                df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
                return df
            except Exception as err:
                print('获取K线获取发生错误! 一秒后重试!', err)

    def place_margin_limit_order_bitfinex(self, symbol, signal, position, margin_balance, position_amount, leverage_rate):
        '''
        根据交易信号和持仓情况自动选择买卖操作 (保证金交易)
        共六种情况:
        1. 空头信号且不持仓 : 开空仓
        2. 空头信号且持有多仓 : 先平多仓再开空仓
        3. 多头信号且不持仓 : 开多仓
        4. 多头信号且持有空仓 : 先平空仓再开多仓
        5. 平仓信号且持有多仓 : 平多仓
        6. 平仓信号且持有空仓 : 平空仓
        :param exchange: 接受一个ccxt.exchange() 对象
        :param symbol: 交易品种
        :param signal: 接受交易信号
        :param position: +1 表示持有多仓, -1 表示持有空仓, 0 表示不持仓
        :param position_amount: 持仓数量
        :param leverage_rate: 杠杆倍数
        :return: order_info
        '''
        while True:
            try:
                if signal == -1 and position == 0:
                    price = self.fetch_ticker(symbol)['bid']
                    amount = margin_balance * leverage_rate / price
                    order_info = self.create_limit_sell_order(symbol, amount, price * 0.98, {'type': 'limit'})
                    order_type = '开空仓'
                    print('下单成功,', symbol, '订单种类:', order_type, '订单价格', price, '订单数量', amount)
                    return order_info

                elif signal == -1 and position == 1:
                    price = self.fetch_ticker(symbol)['bid']
                    amount = position_amount
                    self.create_limit_sell_order(symbol, amount, price * 0.98, {'type': 'limit'})
                    price = self.fetch_ticker(symbol)['bid']
                    amount = margin_balance * leverage_rate / price
                    order_info = self.create_limit_sell_order(symbol, amount, price * 0.98, {'type': 'limit'})
                    order_type = '平多仓后开空仓'
                    print('下单成功,', symbol, '订单种类:', order_type, '订单价格', price, '订单数量', amount)
                    return order_info

                elif signal == 0 and position == 1:
                    price = self.fetch_ticker(symbol)['bid']
                    amount = position_amount
                    order_info = self.create_limit_sell_order(symbol, amount, price * 0.98, {'type': 'limit'})
                    order_type = '平多仓'
                    print('下单成功,', symbol, '订单种类:', order_type, '订单价格', price, '订单数量', amount)
                    return order_info

                elif signal == 0 and position == -1:
                    price = self.fetch_ticker(symbol)['ask']
                    amount = position_amount
                    order_info = self.create_limit_buy_order(symbol, amount, price * 1.02, {'type': 'limit'})
                    order_type = '平空仓'
                    print('下单成功,', symbol, '订单种类:', order_type, '订单价格', price, '订单数量', amount)
                    return order_info

                elif signal == 1 and position == 0:
                    price = self.fetch_ticker(symbol)['ask']
                    amount = margin_balance * leverage_rate / price
                    order_info = self.create_limit_buy_order(symbol, amount, price * 1.02, {'type': 'limit'})
                    order_type = '开多仓'
                    print('下单成功,', symbol, '订单种类:', order_type, '订单价格', price, '订单数量', amount)
                    return order_info

                elif signal == 1 and position == -1:
                    price = self.fetch_ticker(symbol)['ask']
                    amount = position_amount
                    self.create_limit_buy_order(symbol, amount, price * 1.02, {'type': 'limit'})
                    price = self.fetch_ticker(symbol)['ask']
                    amount = margin_balance * leverage_rate / price
                    order_info = self.create_limit_buy_order(symbol, amount, price * 1.02, {'type': 'limit'})
                    order_type = '平空仓后开多仓'
                    print('下单成功,', symbol, '订单种类:', order_type, '订单价格', price, '订单数量', amount)
                    return order_info

                else:
                    return '无交易信号, 未执行任何下单操作'

            except Exception as error:
                leverage_rate *= 0.99
                print('下单失败, 1秒后重新下单, 错误信息 : ', error)

    def bfx_margin_account_equity(self, file_path):
        margin_balance = float(self.private_post_margin_infos()[0]['margin_balance'])
        now_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        f = open(file_path, mode='a', newline='')
        writer = csv.writer(f)
        writer.writerow([now_time, margin_balance])
        f.close()
        print('保证金账户最新资金已添加')

    def trade(self):
        def _stop_loss_signal():
            '''
            Involves I/O操作， 所以用多线程
            :return:
            '''
            global configDoc
            tradeConfigDir = Path().parent / 'data' / 'trade_config'
            configDoc = tradeConfigDir / 'config.txt'
            tradeConfigDirExists: bool = tradeConfigDir.exists()
            configDocExists: bool = configDoc.exists()
            if tradeConfigDirExists and configDocExists and os.stat(configDoc).st_size:
                with open(configDoc, mode='r') as f:
                    try:
                        open_p = float(f.read())
                        stopLossPct = self.params[2]
                        if (self.fetch_ticker(self.symbol)['last'] >= open_p * (
                                1 + stopLossPct / 100) and position == -1) or \
                                (self.fetch_ticker(self.symbol)['last'] <= open_p * (
                                        1 - stopLossPct / 100) and position == 1):
                            return 0
                        else:
                            return None
                    except TypeError as err:
                        print(err)
                        return None
            elif not tradeConfigDirExists and not configDocExists:
                tradeConfigDir.mkdir()
                configDoc.touch()
                return None
            elif not configDocExists and tradeConfigDirExists:
                configDoc.touch()
                return None
            elif configDocExists and not os.stat(configDoc).st_size:
                return None

        while True:
            # 用的信号
            # ///
            signal_generator = Signal.bolling_signal
            # ///
            pool = ThreadPool()
            pool.apply_async(self.fetch_my_balance)
            # Pos
            time.sleep(10)
            pos_result_obj = pool.apply_async(self.privatePostPositions)
            pos = pos_result_obj.get()
            if pos:
                if float(pos[0]['amount']) > 0:
                    position = 1
                    position_amount = abs(float(pos[0]['amount']))
                    print('持仓信息: 当前为多头仓位, 总持仓为', position_amount, pos[0]['symbol'])
                else:
                    position = -1
                    position_amount = abs(float(pos[0]['amount']))
                    print('持仓信息: 当前为空头仓位, 总持仓为', position_amount, pos[0]['symbol'])
            else:
                position = 0
                position_amount = 0
                print('持仓信息: 当前不持仓')
            stop_loss_signal_obj = pool.apply_async(_stop_loss_signal)
            stop_loss_signal = stop_loss_signal_obj.get()
            margin_balance = float(pool.apply_async(self.private_post_margin_infos).get()[0]['margin_balance'])
            break_flag = True
            while break_flag:
                run_time = self.next_run_time(self.timeInterval, initial_min=0)
                sleep(max(0, (run_time - datetime.now()).seconds) - 1)
                while True:
                    if datetime.now() < run_time:
                        continue
                    else:
                        start = time.perf_counter()
                        break
                counter = 0
                while counter < 3:
                    try:
                        df = self.parsed_bitfinex_recent_candles()
                    except ccxt.RequestTimeout as err:
                        print(f'[Error]{err}')
                        counter += 1
                        continue
                    _temp = df[df['candle_begin_time'] == (run_time - timedelta(
                        minutes=int(self.timeInterval.strip('m') if self.timeInterval != '1h' else 60)))]
                    if _temp.empty:
                        counter += 1
                        print('获取数据不包含最新的数据，1s后重新获取')
                    else:
                        break_flag = False
                        counter = 10

            # Signal
            df = df[df['candle_begin_time'] < pd.to_datetime(run_time)]
            df = signal_generator(df, params=self.params)
            signal = df.iloc[-1]['signal']
            if stop_loss_signal == 0:
                signal = 0
            order_info = self.place_margin_limit_order_bitfinex(self.symbol, signal, position, margin_balance,
                                                                             position_amount, self.leverage_rate)
            end = time.perf_counter()
            print(f'Finished in {end - start} seconds')
            print('交易信号', signal)
            print('下单信息: ', order_info)
            print(df.tail())
            if datetime.now().hour == 0 and datetime.now().minute == 0:
                try:
                    self.bfx_margin_account_equity(file_path=Path().parent / 'data' / 'margin balance' /'margin_balance.csv')
                except PermissionError:
                    print('已经打开的文件无法写入,请关闭')
            if (signal == 1 or signal == -1) and not pos:
                with open(configDoc, mode='w') as f:
                    f.write(str(self.fetch_ticker(self.symbol)['last']))
            print('\n=====本次循环结束======\n')

    def nonce(self):
        return f'{time.time()*1e99:0.0f}'

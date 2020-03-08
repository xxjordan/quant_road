from multiprocessing import Pool
import random
import pandas as pd
import numpy as np
import talib
import json
import time


coin = pd.read_hdf('bfx_btc_1min.hdf5')
threshold = 10000000

coin['amount'] = coin['volume'] * coin['close']  # 这里的成交额只能粗略的计算一下
coin['amount_sum'] = coin['amount'].cumsum()
coin['k_line'] = coin['amount_sum'] // threshold


def f1(x):
    x.reset_index(drop=True, inplace=True)
    #     datetime = x.at[0,'datetime']
    datetime = x.at[0, 'k_line']
    open = x.at[0, 'open']
    close = x.iloc[-1]['close']
    high = x['high'].max()
    low = x['low'].min()
    volume = x['volume'].sum()
    k_num = x['k_line'].sum() / x['k_line'][0]
    k_line_list.append(
        {'datetime': datetime, 'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume,
         'k_num': k_num})


k_line_list = []
coin.groupby('k_line').apply(f1)
coin = pd.DataFrame(k_line_list)


def Heiken_ashi(data):
    data['H_close'] = (data['open'] + data['close'] + data['high'] + data['low']) / 4
    data['H_open'] = (data['open'].shift() + data['open']) / 2
    data['H_high'] = data[['H_open', 'H_close', 'high']].max(axis=1)
    data['H_low'] = data[['H_open', 'H_close', 'low']].min(axis=1)

    return data


coin = Heiken_ashi(coin)
coin_bak = coin.copy()


class Genetic_algo():
    def __init__(self, algo_func, score_func, pop_size, pc, pm, generations, threshold, chrom_dict):
        '''
        :param algo_func: 算法函数
        :param pop_size: 种群数量
        :param pc: 交配概率
        :param pm: 变异概率
        :param chrom_dict: 染色体函数
        :param score_func: 评价函数
        :param threshold: 阈值，删掉评价小于阈值的种群
        :param generations: 繁殖代数
        '''
        self.algo_func = algo_func
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.generations = generations
        self.chrom_dict = chrom_dict
        self.pop_list = self.init_pop_list(pop_size)
        self.max_result = []
        self.score_func = score_func
        self.threshold = threshold

    def init_pop_list(self, num):
        pop_list = []
        for i in range(num):
            pop_dict = {}
            for k, v in self.chrom_dict.items():
                pop_dict[k] = v()
            pop_dict['index'] = i
            pop_list.append(pop_dict)
        return pop_list

    def run(self):
        for _ in range(self.generations):
            pool = Pool(11)
            print('第%s轮'%(_+1))
            with open('第%s轮.json'%(_+1), 'w') as f:
                json.dump(self.pop_list, f)
            results = pool.map(self.algo_func, self.pop_list)
            pool.close()
            pool.join()
            # results = self.algo_func(self.pop_list[0])
            max_score = 0
            max_result = 0
            self.pop_list = []
            i = 0
            for idx, result in results:
                score = self.score_func(result)
                if score > self.threshold or score == -1:
                    params = result[3]
                    params['index'] = i
                    self.pop_list.append(params.copy())
                    i+=1
                if score == -1:
                    self.max_result.append(
                        {'result': result.copy(), 'score': -1})
                if score > max_score:
                    max_score = score.copy()
                    max_result = result.copy()
            self.max_result.append(
                {'result': max_result, 'score': max_score})
            if len(self.pop_list) < 2:
                return self.pop_list, self.max_result
            self.pair()
        return self.pop_list, self.max_result

    def crossover(self, x, y):
        for k, v in x.items():
            if random.random() <= self.pc and k != 'index':
                x[k], y[k] = y[k], x[k]
        return x, y

    def mutation(self, x):
        for k, v in x.items():
            if random.random() <= self.pm and k != 'index':
                x[k] = self.chrom_dict[k]()
        return x

    def pair(self):
        pop_list = []
        n = len(self.pop_list) // 2
        for i in range(n):
            x_, y_ = random.sample(self.pop_list, 2)
            self.pop_list.remove(x_)
            self.pop_list.remove(y_)
            x = x_.copy()
            y = y_.copy()
            x, y = self.crossover(x, y)
            x['index'] = i * 2
            y['index'] = i * 2 + 1
            x = self.mutation(x)
            y = self.mutation(y)
            pop_list.append(x)
            pop_list.append(y)

        self.pop_list = pop_list

def score_func(result):
    curve = result[0]
    max_drowdown = result[1]
    max_curve = result[2]
    if curve > 1.2:
        if max_drowdown < 0.1:
            return -1
        elif max_drowdown < 0.18:
            return max_curve - max_drowdown + curve + 0.5
        return max_curve - max_drowdown + curve

    else:
        return 0



def algo_main(params):
    coin = coin_bak.copy()
    timeperiod = params.get('timeperiod')
    ma = params.get('ma')
    dx0 = params.get('dx0')
    dx1 = params.get('dx1')
    dx2 = params.get('dx2')
    dx3 = params.get('dx3')
    dx = [dx0,dx1,dx2,dx3]
    dx.sort()
    dx0 = dx[0]
    dx1 = dx[1]
    dx2 = dx[2]
    dx3 = dx[3]
    args0 = params.get('args0')
    args1 = params.get('args1')
    args2 = params.get('args2')
    args3 = params.get('args3')
    args4 = params.get('args4')
    args = [args0, args1, args2, args3,args4]
    args.sort()
    args0 = args[0]
    args1 = args[1]
    args2 = args[2]
    args3 = args[3]
    args4 = args[4]
    coin['ADX'] = talib.DX(coin['H_high'], coin['H_low'], coin['H_close'], timeperiod=timeperiod)

    def x1(data, r):
        x = pd.Series(data).quantile(r) * 1
        return x

    coin.loc[coin['ADX'] < dx0, 'x'] = coin['H_high'].rolling(ma).apply(x1, args=(1 - args0,))
    coin.loc[(coin['ADX'] >= dx0) & (coin['ADX'] < dx1), 'x'] = coin['H_high'].rolling(ma).apply(x1, args=(
        1 - args1,))
    coin.loc[(coin['ADX'] >= dx1) & (coin['ADX'] < dx2), 'x'] = coin['H_high'].rolling(ma).apply(x1, args=(
        1 - args2,))
    coin.loc[(coin['ADX'] >= dx2) & (coin['ADX'] < dx3), 'x'] = coin['H_high'].rolling(ma).apply(x1, args=(
        1 - args3,))
    coin.loc[(coin['ADX'] >= dx3), 'x'] = coin['H_high'].rolling(ma).apply(x1, args=(1 - args4,))

    coin.loc[coin['ADX'] < dx0, 'y'] = coin['H_low'].rolling(ma).apply(x1, args=(args0,))
    coin.loc[(coin['ADX'] >= dx0) & (coin['ADX'] < dx1), 'y'] = coin['H_low'].rolling(ma).apply(x1, args=(args1,))
    coin.loc[(coin['ADX'] >= dx1) & (coin['ADX'] < dx2), 'y'] = coin['H_low'].rolling(ma).apply(x1, args=(args2,))
    coin.loc[(coin['ADX'] >= dx2) & (coin['ADX'] < dx3), 'y'] = coin['H_low'].rolling(ma).apply(x1, args=(args3,))
    coin.loc[(coin['ADX'] >= dx3), 'y'] = coin['H_low'].rolling(ma).apply(x1, args=(args4,))
    coin['pos'] = 0
    coin.loc[coin['x'] < coin['close'], 'pos'] = 1
    coin.loc[coin['y'] > coin['close'], 'pos'] = -1

    coin.reset_index(drop=True, inplace=True)

    coin['position'] = coin['pos'].shift()

    coin['signal'] = np.nan
    coin.loc[(coin['position'] != 1) & (coin['position'].shift(-1) == 1), 'signal'] = 1
    coin.loc[(coin['position'] != -1) & (coin['position'].shift(-1) == -1), 'signal'] = -1
    coin.loc[(coin['position'] != 0) & (coin['position'].shift(-1) == 0), 'signal'] = 0
    coin['rate'] = (coin['close'] - coin['close'].shift()) / coin['close'].shift()
    coin['chg_m'] = coin['rate'] * coin['position'] * 1
    coin['stock_curve'] = coin['close'] / coin.at[0, 'open']
    coin['curve'] = (1 + coin['chg_m'])
    coin.loc[(coin['signal'] == 1) | (coin['signal'] == -1) | (coin['signal'] == 0), 'curve'] = coin[
                                                                                                    'curve'] - 0.00075 * 3
    coin['curve'] = coin['curve'].cumprod()

    coin['curve_max'] = coin['curve'].cummax()
    coin['drowdown'] = (coin['curve_max'] - coin['curve']) / coin['curve_max']

    print(params.get('index'),coin.iloc[-1]['curve'], coin['drowdown'].max(),coin['curve'].max(),params)

    params['dx0'] = dx0
    params['dx1'] = dx1
    params['dx2'] = dx2
    params['dx3'] = dx3
    params['args0'] = args0
    params['args1'] = args1
    params['args2'] = args2
    params['args3'] = args3
    params['args4'] = args4

    return params.get('index'),[coin.iloc[-1]['curve'], coin['drowdown'].max(),coin['curve'].max(),params]


def main():
    def timeperiod_func():
        return random.randint(3, 60)

    def ma_func():
        return random.randint(3, 99)

    def dx_func():
        return random.randint(1, 38)

    def args_func():
        return random.random() / 2
    g = Genetic_algo(algo_main,score_func , 1000, 0.5, 0.05, 10, 2, {'timeperiod':timeperiod_func,
                                 'ma':ma_func,
                                 'dx0':dx_func,
                                 'dx1':dx_func,
                                 'dx2':dx_func,
                                 'dx3':dx_func,
                                 'args0':args_func,
                                 'args1':args_func,
                                 'args2':args_func,
                                 'args3':args_func,
                                 'args4':args_func,
                                 })

    pop_list, max_result = g.run()

    print(pop_list)
    print('_________')
    print(max_result)
    with open('best.json','w') as f:
        json.dump(max_result,f)
    with open('finall.json','w') as f:
        json.dump(pop_list,f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('用时',time.time() - start_time)
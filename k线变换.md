# 成交量等分k线
我们常见的k线都是时间等分的普通k线，我们也可以对K线进行其他处理  
例如：以固定成交量等分的k线，以固定成交额等分的k线，以固定交易笔数等分的k线。  

# 成交量等分k线
画成交额等分的k线需用用到逐笔数据，我手里现在只有最小粒度为1min的数据，我就以这样的数据进行代码实现。  
代码如下：  
```
coin = pd.read_csv('bfx_btcusd.csv',names=['timestamp','open','high','low','close','volume'])
coin['datetime'] = pd.to_datetime(coin['timestamp'],unit='ms')

coin = coin[['datetime','open','high','low','close','volume']]
coin['datetime'] = coin.index
coin.dropna(inplace=True)

threshold = 1000  # 设定每根k线的量为1000
coin['volume_sum'] = coin['volume'].cumsum()
coin['k_line'] = coin['volume_sum']//threshold
def f(x):
    x.reset_index(drop=True,inplace=True)
    k_line = x.at[0,'k_line']
    open = x.at[0,'open']
    close = x.iloc[-1]['close']
    high = x['high'].max()
    low = x['low'].min()
    k_line_list.append({'k_line':k_line,'open':open,'high':high,'low':low,'close':close})
k_line_list=[]
coin.groupby('k_line').apply(f)
coin = pd.DataFrame(k_line_list)
```

# 成交额等分k线
和成交量等分k线原理类似，直接上代码：
```
coin = pd.read_csv('bfx_btcusd.csv',names=['timestamp','open','high','low','close','volume'])
coin['datetime'] = pd.to_datetime(coin['timestamp'],unit='ms')

coin = coin[['datetime','open','high','low','close','volume']]
coin['datetime'] = coin.index
coin.dropna(inplace=True)

coin['amount'] = coin['volume'] * coin['close'] # 这里的成交额只能粗略的计算一下
coin['amount_sum'] = coin['amount'].cumsum()
threshold = 1000  # 设定每根k线的量为1000
coin['k_line'] = coin['amount_sum']//threshold
def f(x):
    x.reset_index(drop=True,inplace=True)
    k_line = x.at[0,'k_line']
    open = x.at[0,'open']
    close = x.iloc[-1]['close']
    high = x['high'].max()
    low = x['low'].min()
    k_line_list.append({'k_line':k_line,'open':open,'high':high,'low':low,'close':close})
k_line_list=[]
coin.groupby('k_line').apply(f)
coin = pd.DataFrame(k_line_list)
```
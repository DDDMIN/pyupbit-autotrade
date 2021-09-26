import time
import pyupbit
import datetime
import requests
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import talib 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from sklearn.preprocessing import MinMaxScaler
import time
import pyupbit
import datetime
import requests
import tensorflow as tf
import random as python_random
from pytrends.request import TrendReq
import datetime
import schedule

access = ""
secret = ""
myToken = ""

def deeplearning():
    #날짜 설정
    secs = time.time()
    tm = time.localtime(secs)

    start_date = '2017-01-01'
    end_date = time.strftime('%Y-%m-%d', tm)

    start_compare = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_compare = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_diff = (end_compare - start_compare).days

    #비트코인 티커 설정
    tick = "KRW-BTC"

    #Upbit 데이터 불러오기
    df = pyupbit.get_ohlcv(tick, count = date_diff, period = 1)
    df.to_csv('btc.csv')
    eth_df = pyupbit.get_ohlcv("KRW-ETH", count = date_diff, period = 1)
    eth_df.to_csv('eth.csv')

    #Yahoo Finance에서 데이터 불러오기
    snp500_df = pdr.get_data_yahoo('^GSPC', start = start_date)
    usd_df = pdr.get_data_yahoo('DX-Y.NYB', start = start_date)
    vix_df = pdr.get_data_yahoo('^VIX', start = start_date)
    gld_df = pdr.get_data_yahoo('GLD', start = start_date)
    m2_df = pdr.get_data_fred('M2', start = start_date)

    #구글 트랜드 데이터 불러오기
    keyword = 'Bitcoin'
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=[keyword], timeframe= start_date + ' ' + end_date)
    btc_trend_df = pytrend.interest_over_time()

    #데이터 불러오기
    df['next_price'] = df['close'].shift(-1) #shift(-1) 다음날 종가
    df['next_rtn'] = df['close'] / df['open'] -1  #다음날 수익률 예측하도록 문제를 정의
    df['log_return'] = np.log(1 + df['close'].pct_change())
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

    #1.RA : Standard deviation rolling average
    # Moving Average 이동평균
    df['MA5'] = talib.SMA(df['close'],timeperiod=5)
    df['MA10'] = talib.SMA(df['close'],timeperiod=10)
    df['MA20'] = talib.SMA(df['close'],timeperiod=20)
    df['RASD5'] = talib.SMA(talib.STDDEV(df['close'], timeperiod=5, nbdev=1),timeperiod=5)
    df['RASD10'] = talib.SMA(talib.STDDEV(df['close'], timeperiod=5, nbdev=1),timeperiod=10)

    #2.MACD : Moving Average Convergence/Divergence
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd 

    # Momentum Indicators
    #3.CCI : Commodity Channel Index
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    # Volatility Indicators 

    #4.ATR : Average True Range
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    #5.BOLL : Bollinger Band
    upper, middle, lower = talib.BBANDS(df['close'],timeperiod=20,nbdevup=2,nbdevdn=2,matype=0)
    df['ub'] = upper
    df['middle'] = middle
    df['lb'] = lower

    #7.MTM1 
    df['MTM1'] = talib.MOM(df['close'], timeperiod=1)

    #7.MTM3
    df['MTM3'] = talib.MOM(df['close'], timeperiod=3)

    #8.ROC : Rate of change : ((price/prevPrice)-1)*100
    df['ROC'] = talib.ROC(df['close'], timeperiod=60)

    #9.WPR : william percent range (Williams' %R)
    df['WPR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

    #종가 Column을 각 종목명으로 변경, 햇갈리지 않게
    snp500_df = snp500_df.loc[:,['Close']].copy()
    snp500_df.rename(columns={'Close':'S&P500'},inplace=True)
    usd_df = usd_df.loc[:,['Close']].copy()
    usd_df.rename(columns={'Close':'USD'},inplace=True)
    btc_trend_df = btc_trend_df.loc[:,['Bitcoin']].copy()
    btc_trend_df.rename(columns={'Bitcoin':'BTC_Trend'},inplace=True)
    vix_df = vix_df.loc[:,['Close']].copy()
    vix_df.rename(columns={'Close':'VIX'},inplace=True)
    gld_df = gld_df.loc[:,['Close']].copy()
    gld_df.rename(columns={'Close':'GLD'},inplace=True)
    #m2는 이미 column이 M2로 되어있어 생략
    eth_df = eth_df.loc[:,['close']].copy()
    eth_df.rename(columns={'close':'ETH'},inplace=True)

    #row index 정리
    snp500_df_index = list(snp500_df.index)
    sub_index = []
    for index in snp500_df_index:
        sub_index.append(index + datetime.timedelta(hours=9))
    snp500_df['Date'] = sub_index
    sub_index = []
    snp500_df = snp500_df.set_index('Date')

    usd_df_index = list(usd_df.index)
    sub_index = []
    for index in usd_df_index:
        sub_index.append(index + datetime.timedelta(hours=9))
    usd_df['Date'] = sub_index
    sub_index = []
    usd_df = usd_df.set_index('Date')

    btc_trend_df_index = list(btc_trend_df.index)
    sub_index = []
    for index in btc_trend_df_index:
        sub_index.append(index + datetime.timedelta(hours=9))
    btc_trend_df['date'] = sub_index
    sub_index = []
    btc_trend_df = btc_trend_df.set_index('date')

    vix_df_index = list(vix_df.index)
    sub_index = []
    for index in vix_df_index:
        sub_index.append(index + datetime.timedelta(hours=9))
    vix_df['date'] = sub_index
    sub_index = []
    vix_df = vix_df.set_index('date')

    gld_df_index = list(gld_df.index)
    sub_index = []
    for index in gld_df_index:
        sub_index.append(index + datetime.timedelta(hours=9))
    gld_df['date'] = sub_index
    sub_index = []
    gld_df = gld_df.set_index('date')

    m2_df_index = list(m2_df.index)
    sub_index = []
    for index in m2_df_index:
        sub_index.append(index + datetime.timedelta(hours=9))
    m2_df['date'] = sub_index
    sub_index = []
    m2_df = m2_df.set_index('date')

    #데이터 날짜 기준으로 결합
    df = df.join(snp500_df,how='left')
    df = df.join(usd_df,how='left')
    df = df.join(btc_trend_df,how='left')
    df = df.join(vix_df,how='left')
    df = df.join(gld_df,how='left')
    df = df.join(eth_df,how='left')
    df = df.join(m2_df,how='left')

    #결측치는 이전값으로 채우기
    df = df.fillna(method = 'pad')
    #이동평균으로 발생하는 결측치 삭제
    df = df.dropna()

    # feature list
    # feature_list = ['Adj Close', 'log_return', 'CCI','next_price']
    # 볼린저 밴드와 MACD를 어떻게 활용해야할까? 음. 아님 그냥 그대로 사용하는 건가?
    feature1_list = ['open','high','low','close','volume','log_return']
    #feature2_list = ['RASD5','RASD10','ub','lb','CCI','ATR','MACD','MA5','MA10','MTM1','MTM3','ROC','WPR']
    feature2_list = ['MA5', 'MA10', 'ROC']
    #feature3_list = ['S&P500', 'USD', 'BTC_Trend', 'VIX', 'GLD', 'ETH', 'M2']
    feature3_list = ['S&P500', 'USD', 'GLD']
    # feature4_list = ['next_price']
    feature4_list = ['next_rtn']

    all_features = feature1_list + feature2_list + feature3_list + feature4_list

    phase_flag = '3'
    #학습, 검증, 테스트 각각 2년 3개월 3개월
    if phase_flag == '1' :
        train_from = '2010-01-04'
        train_to = '2012-01-01'

        val_from = '2012-01-01'
        val_to = '2012-04-01'

        test_from = '2012-04-01'
        test_to = '2012-07-01'

    elif phase_flag == '2' :
        train_from = '2012-07-01'
        train_to = '2014-07-01'

        val_from = '2014-07-01'
        val_to = '2014-10-01'

        test_from = '2014-10-01'
        test_to = '2015-01-01'
        
    else : 
        train_from = start_date
        train_to = '2021-01-01'

        val_from = '2021-01-01'
        val_to = '2021-09-01'

        test_from = '2021-09-01'
        test_to = end_date

    # train / validation / testing 데이터셋 분리
    train_df  = df.loc[train_from:train_to,all_features].copy()
    val_df = df.loc[val_from:val_to,all_features].copy()
    test_df   = df.loc[test_from:test_to,all_features].copy()

    #최대최소 정규화
    def min_max_normal(tmp_df):
        eng_list = []
        sample_df = tmp_df.copy()
        for x in all_features:
            if x in feature4_list : #feature중 next price는 예측할 대상이니까 제외
                continue
            series = sample_df[x].copy() #data field를 카피해놓음(원본삭제방지)
            values = series.values #series변수.values 호출하면 values 값만 따로 array로 확인 가능
            values = values.reshape((len(values), 1)) #.reshape(x,y) x행, y열로 재배치
            # train the normalization
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler = scaler.fit(values)
    #         print('columns : %s , Min: %f, Max: %f' % (x, scaler.data_min_, scaler.data_max_))
            # normalize the dataset and print
            normalized = scaler.transform(values)
            new_feature = '{}_normal'.format(x)#feature_normal이라는 column새로 만들기
            eng_list.append(new_feature)
            sample_df[new_feature] = normalized
        return sample_df, eng_list

    train_sample_df, eng_list =  min_max_normal(train_df)
    val_sample_df, eng_list =  min_max_normal(val_df)
    test_sample_df, eng_list = min_max_normal(test_df)

    #LSTM Model 훈련데이터 구분하기
    num_step = 5
    num_unit = 200
    #학습 데이터와 레이블 데이터 구분
    def create_dateset_binary(data, feature_list, step, n):
        '''
        다음날 시종가 수익률 라벨링.
        '''
        train_xdata = np.array(data[feature_list[0:n]])#불러온 데이터의 n번째 행까지 train 데이터로 저장
        # 가장 뒤 n step을 제외하기 위해. 왜냐하면 학습 input으로는 어차피 10개만 주려고 하니깐.
        m = np.arange(len(train_xdata) - step)#np.arange(n) 0부터 n-1까지 1간격으로 array 생성
        #     np.random.shuffle(m)  # shufflee은 빼자.
        x, y = [], []
        for i in m:
            a = train_xdata[i:(i+step)]#train data는 i에서 i+step-1 까지 데이터
            x.append(a)
        x_batch = np.reshape(np.array(x), (len(m), step, n))#array를 x,y,z형태의 3차원 데이터로 변환
        
        train_ydata = np.array(data[[feature_list[n]]])#이경우 feature list4번(nxt return)임
        # n_step 이상부터 답을 사용할 수 있는거니깐. 
        for i in m + step :
            next_rtn = train_ydata[i][0]
            if next_rtn > 0 :
                label = 1
            else :
                label = 0
            y.append(label)
        y_batch = np.reshape(np.array(y), (-1,1))
        return x_batch, y_batch

    eng_list = eng_list + feature4_list
    n_feature = len(eng_list)-1
    # LSTM할때 사용했던 소스코드.
    x_train, y_train = create_dateset_binary(train_sample_df[eng_list], eng_list, num_step, n_feature)
    x_val, y_val = create_dateset_binary(val_sample_df[eng_list], eng_list, num_step, n_feature)
    x_test, y_test = create_dateset_binary(test_sample_df[eng_list], eng_list, num_step, n_feature)

    from tensorflow.keras.utils import to_categorical

    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)

    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Input, Dense, LSTM
    from tensorflow.keras.layers import Activation, BatchNormalization
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras import backend as K
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping

    # LSTM 모델을 생성한다.
    K.clear_session()
    input_layer = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
    layer_lstm_1 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.1))(input_layer)
    layer_lstm_1 = BatchNormalization()(layer_lstm_1)
    layer_lstm_2 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.1))(layer_lstm_1)
    layer_lstm_2 = Dropout(0.25)(layer_lstm_2)
    layer_lstm_3 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.1))(layer_lstm_2)
    layer_lstm_3 = BatchNormalization()(layer_lstm_3)
    layer_lstm_4 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.1))(layer_lstm_3)
    layer_lstm_4 = Dropout(0.25)(layer_lstm_4)
    layer_lstm_5 = LSTM(num_unit , recurrent_regularizer = regularizers.l2(0.01))(layer_lstm_4)#마지막에는 return_sequences 빼줌
    layer_lstm_5 = BatchNormalization()(layer_lstm_5)
    output_layer = Dense(2, activation='sigmoid')(layer_lstm_5)

    #LSTM 모델 컴파일
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(patience = 20)

    history = model.fit(x_train,
                        y_train,
                        epochs=200, #반복 훈련 횟수
                        batch_size=64, #배치 size
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping]
                    )

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    predicted = model.predict(x_test)#테스트 데이터로 모델의 예측값 출력
    y_pred = np.argmax(predicted, axis = 1)#
    Y_test = np.argmax(y_test, axis = 1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)

    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    TPR = float(tp)/(float(tp)+float(fn))
    FPR = float(fp)/(float(fp)+float(tn))
    accuracy = round((float(tp) + float(tn))/(float(tp) +
                                            float(fp) + float(fn) + float(tn)), 3)
    specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
        (float(tp)+float(fp))
        * (float(tp)+float(fn))
        * (float(tn)+float(fp))
        * (float(tn)+float(fn))
    ), 3)

    f_output = open('binary_lstm_open_close_phase3_dropout_batch_Normal_3단계 test.txt', 'a')
    f_output.write('=======\n')
    f_output.write('{}epochs_{}batch\n'.format(
        20, 10))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()

    # 3단계 
    lstm_book_df = test_sample_df[['close','next_rtn']].copy()
    # ### 이 문제에 있어서 Series와 DataFrame의 차이는 뭐지?
    t1 = pd.DataFrame(data = y_pred,columns=['position'],index = lstm_book_df.index[5:])
    lstm_book_df = lstm_book_df.join(t1,how='left')
    lstm_book_df.fillna(0,inplace=True)
    lstm_book_df['ret'] = lstm_book_df['close'].pct_change()
    lstm_book_df['lstm_ret'] = lstm_book_df['next_rtn'] * lstm_book_df['position'].shift(1)
    lstm_book_df['lstm_cumret'] = (lstm_book_df['lstm_ret'] + 1).cumprod()
    lstm_book_df['bm_cumret'] = (lstm_book_df['ret'] + 1).cumprod()

    #백테스팅
    #back testing
    historical_max = lstm_book_df['close'].cummax()
    daily_drawdown = lstm_book_df['close'] / historical_max - 1.0
    historical_dd = daily_drawdown.cummin()
    historical_dd.plot()

    #밴치마크 백테스팅
    CAGR_B = lstm_book_df.loc[lstm_book_df.index[-1],'bm_cumret'] ** (365./len(lstm_book_df.index)) -1
    Sharpe = np.mean(lstm_book_df['ret']) / np.std(lstm_book_df['ret']) * np.sqrt(365.)
    VOL = np.std(lstm_book_df['ret']) * np.sqrt(365.)
    MDD = historical_dd.min()
    print('CAGR_B : ',round(CAGR_B*100,2),'%')
    print('Sharpe : ',round(Sharpe,2))
    print('VOL : ',round(VOL*100,2),'%')
    print('MDD : ',round(-1*MDD*100,2),'%')

    #LSTM 백테스팅
    CAGR_L = lstm_book_df.loc[lstm_book_df.index[-1],'lstm_cumret'] ** (365./len(lstm_book_df.index)) -1
    if np.std(lstm_book_df['lstm_ret']) != 0:
        Sharpe = np.mean(lstm_book_df['lstm_ret']) / np.std(lstm_book_df['lstm_ret']) * np.sqrt(365.)
    else:
        Sharpe = 0
    VOL = np.std(lstm_book_df['lstm_ret']) * np.sqrt(365.)
    MDD = historical_dd.min()
    position = t1.iloc[-1]['position']
    print('CAGR_L : ',round(CAGR_L*100,2),'%')
    print('Sharpe : ',round(Sharpe,2))
    print('VOL : ',round(VOL*100,2),'%')
    print('MDD : ',round(-1*MDD*100,2),'%')

    print('Today Position : ', position)

    return CAGR_L, CAGR_B, position

def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )
    print(response)

def get_target_price(ticker, k):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df_target = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    target_price = df_target.iloc[0]['close'] + (df_target.iloc[0]['high'] - df_target.iloc[0]['low']) * k
    return target_price

def get_ror(k=0.5):
    """k값 백테스팅"""
    df_ror = pyupbit.get_ohlcv("KRW-BTC", count=7)
    df_ror['range'] = (df_ror['high'] - df_ror['low']) * k
    df_ror['target'] = df_ror['open'] + df_ror['range'].shift(1)
    fee = 0.0005
    df_ror['ror'] = np.where(df_ror['high'] > df_ror['target'],
                         df_ror['close'] / df_ror['target'] - fee,
                         1)
    ror = df_ror['ror'].cumprod()[-2]
    return ror

def get_start_time(ticker):
    """시작 시간 조회"""
    df_time = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df_time.index[0]
    return start_time

def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_current_price(ticker):
    """현재가 조회"""
    return pyupbit.get_orderbook(tickers=ticker)[0]["orderbook_units"][0]["ask_price"]

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")

# 시작 메세지 슬랙 전송
post_message(myToken,"#crypto", "autotrade start")

#딥러닝 백테스트 결과 cl, 밴치마크 백테스트 결과 cb, 매수, 매도 포지션
cl, cb, pos = deeplearning()

while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-BTC")
        end_time = start_time + datetime.timedelta(days=1)
 
        #오늘 09:00시 딥러닝 업데이트
        if start_time < now < start_time + datetime.timedelta(minutes=1):
            cl, cb, pos = deeplearning()

        #오늘 09:01~ 내일 08:59까지 모니터링, 매수
        if start_time + datetime.timedelta(minutes=1) < now < end_time - datetime.timedelta(minutes=1):
            ror_max = 0
            best_k = 0
            for k in np.arange(0.1, 1.0, 0.1):
                ror = get_ror(k)
                if ror > ror_max:
                    ror_max = ror
                    best_k = k
            
            target_price = get_target_price("KRW-BTC", best_k)
            current_price = get_current_price("KRW-BTC")
            if cl > cb and pos == 1 and target_price < current_price:
                krw = get_balance("KRW")
                if krw > 5000:
                    buy_result = upbit.buy_market_order("KRW-BTC", krw*0.9995)
                    post_message(myToken,"#crypto", "BTC buy : " +str(buy_result))
                    print('buy coin')
            else:
                print('hold')
        #내일 08:59~09:00 사이에 매도
        else:
            btc = get_balance("BTC")
            if btc > 0.00008:
                sell_result = upbit.sell_market_order("KRW-BTC", btc*0.9995)
                post_message(myToken,"#crypto", "BTC sell : " +str(sell_result))
                print('sell coin')
        time.sleep(1)
    except Exception as e:
        print(e)
        post_message(myToken,"#crypto", e)
        time.sleep(1)


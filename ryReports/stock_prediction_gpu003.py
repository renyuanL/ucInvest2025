
# stock_prediction_gpu03.py

'''
股票價格預測與投資策略之量化分析研究報告

一、研究背景與目的

隨著金融科技（FinTech）與人工智慧（AI）技術快速發展，利用機器學習（Machine Learning, ML）預測股票價格的準確性已逐步提高。本報告旨在探討透過量化模型，包括線性回歸（Linear Regression）、隨機森林（Random Forest）、XGBoost、長短期記憶神經網路（Long Short-Term Memory, LSTM）、卷積神經網路（Convolutional Neural Network, CNN）與變換器（Transformer），針對NASDAQ100指數股票的價格進行預測，並比較不同特徵組合與投資策略之效果。

二、資料處理與特徵工程

本研究採用NASDAQ100指數股票逐分鐘資料，時間範圍集中於美股交易時段（09:30至16:00）。資料處理流程包括資料載入、日期時間（datetime）轉換與缺值處理。

三、模型與訓練方法

模型建構部分，深度學習模型採用PyTorch框架實現，分別介紹如下：

1. LSTMRegressor：利用LSTM層捕捉長期依賴性。
2. CNN1DRegressor：透過一維卷積（Conv1D）捕捉局部特徵。
3. SimpleTransformerRegressor：透過Transformer編碼器處理序列數據。

模型評估子函數 evaluate\_models 執行流程包括統計模型與深度學習模型的訓練與評估，最終提供模型準確率。

四、主流程執行與結果整合

主流程子函數 run 負責執行股票數據的特徵工程、模型訓練與評估，具體步驟包括資料載入、特徵工程、模型訓練與評估，並儲存結果。

五、投資策略回測函式說明

投資策略回測函式包括三種不同策略：

1. 全進全出策略（All-in/All-out）：

   * 根據模型預測序列決定買入（預測值>0）或賣出（預測值<0）。
   * 計算最終資產、策略總收益率以及資產價值變化的曲線。

2. 買入持有策略（Buy and Hold）：

   * 期初以全部資金買入股票後持有至期末。
   * 計算最終資產、策略總收益率與資產價值變化的曲線。

3. 定時定額策略（Dollar-Cost Averaging, DCA）：

   * 在整個投資期間內平均分配投資資金，於每個時間點以固定金額買入股票。
   * 計算最終資產、策略總收益率以及資產價值變化的曲線。

六、模型績效評估與投資策略

模型評估以預測準確率（Accuracy）為指標，投資策略則包括全進全出、買入持有與定時定額策略。

七、結論與未來研究方向

透過整合多種機器學習模型與豐富特徵，本研究驗證了AI技術在股票預測與投資策略設計之潛力，未來可納入更多市場資訊並測試模型穩健性。

'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import talib
from tqdm import tqdm
#from pathlib import Path

# ====================== 全局參數 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用裝置：", DEVICE)

DATAROOTPATH = "C:/_ryDatasets/nasdaq100_monthly/" #nasdaq100_{year}{month}.h5")
# 檢查資料根目錄是否存在
if not pd.io.common.file_exists(DATAROOTPATH):
    raise FileNotFoundError(f"資料根目錄不存在：{DATAROOTPATH}")
# 如果資料根目錄不存在，則拋出錯誤
else:
    print(f"資料根目錄 {DATAROOTPATH} 存在，繼續執行...")

# ====================== 1. 資料處理 ======================
def load_stock_data(year='2015', month='01', 
                   path_template= DATAROOTPATH + "nasdaq100_{year}{month}.h5"
                   ):
    file_path = path_template.format(year=year, month=month)
    df = pd.read_hdf(file_path, key="minute_data")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['TimeBarStart'])
    df = df[(df['datetime'].dt.time >= pd.to_datetime('09:30').time()) &
            (df['datetime'].dt.time <= pd.to_datetime('16:00').time())]
    df.sort_values(['Ticker', 'datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['LastTradePrice'] = df['LastTradePrice'].ffill()
    df['Volume'] = df['Volume'].fillna(0)
    return df

# ====================== 2. 特徵工程 ======================
def make_features(df, config, lag_n=5):
    feats = []
    for lag in range(1, lag_n + 1):
        df[f'lag_{lag}'] = df['LastTradePrice'].shift(lag)
        feats.append(f'lag_{lag}')
    if config.get('use_volume_lag', False):
        for lag in range(1, lag_n + 1):
            df[f'vol_lag_{lag}'] = df['Volume'].shift(lag)
            feats.append(f'vol_lag_{lag}')
    if config.get('use_ta', False):
        df['RSI'] = talib.RSI(df['LastTradePrice'], timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['LastTradePrice'], 12, 26, 9)
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['LastTradePrice'], 20, 2, 2, 0)
        feats += ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper_band', 'middle_band', 'lower_band']
    if config.get('use_more', False):
        df['EMA_20'] = talib.EMA(df['LastTradePrice'], 20)
        df['EMA_50'] = talib.EMA(df['LastTradePrice'], 50)
        df['EMA_100'] = talib.EMA(df['LastTradePrice'], 100)
        feats += ['EMA_20', 'EMA_50', 'EMA_100']
        for col in ['TotalTrades', 'VolumeWeightPrice', 'MinSpread', 'MaxSpread']:
            df[col] = df.get(col, pd.Series([0]*len(df))).fillna(0)
        df['VolumeValue'] = df['Volume'] * df['VolumeWeightPrice']
        feats += ['TotalTrades', 'VolumeWeightPrice', 'VolumeValue', 'MinSpread', 'MaxSpread']
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    feats += ['hour', 'minute']
    df['y'] = df['LastTradePrice'].shift(-1) - df['LastTradePrice']
    df['y'] = np.where(df['y'] > 0, 1, -1)
    df = df.dropna()
    return df, feats

# ====================== 3. 模型 ======================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class CNN1DRegressor(nn.Module):
    def __init__(self, input_size, channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=2)
        self.fc = nn.Linear((input_size-1)*channels, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SimpleTransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# ====================== 4. 訓練與評估 ======================
def evaluate_models(X_train, y_train, X_test, y_test, epochs=10):
    results, y_preds, y_tests = {}, {}, {}

    # scikit-learn 傳統模型
    base_models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=0),
    }
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = (np.sign(y_pred) == np.sign(y_test)).mean()
        results[name] = acc
        y_preds[name], y_tests[name] = y_pred, y_test

    # LSTM
    X_train_lstm = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_test_lstm  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(DEVICE)
    lstm = LSTMRegressor(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(lstm.parameters(), lr=0.01)
    crit = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(lstm(X_train_lstm), y_train_torch)
        loss.backward(); opt.step()
    with torch.no_grad():
        y_pred = lstm(X_test_lstm).view(-1).cpu().numpy()
    acc = (np.sign(y_pred) == np.sign(y_test)).mean()
    results["LSTM"], y_preds["LSTM"], y_tests["LSTM"] = acc, y_pred, y_test

    # CNN
    X_train_cnn = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_test_cnn  = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    cnn = CNN1DRegressor(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(cnn.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(cnn(X_train_cnn), y_train_torch)
        loss.backward(); opt.step()
    with torch.no_grad():
        y_pred = cnn(X_test_cnn).view(-1).cpu().numpy()
    acc = (np.sign(y_pred) == np.sign(y_test)).mean()
    results["CNN"], y_preds["CNN"], y_tests["CNN"] = acc, y_pred, y_test

    # Transformer
    X_train_tf = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_test_tf  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    transformer = SimpleTransformerRegressor(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(transformer.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(transformer(X_train_tf), y_train_torch)
        loss.backward(); opt.step()
    with torch.no_grad():
        y_pred = transformer(X_test_tf).view(-1).cpu().numpy()
    acc = (np.sign(y_pred) == np.sign(y_test)).mean()
    results["Transformer"], y_preds["Transformer"], y_tests["Transformer"] = acc, y_pred, y_test

    # Random Guess baseline
    y_random = np.random.choice([-1, 1], size=len(y_test))
    results["RandomGuess"] = (y_random == y_test).mean()
    y_preds["RandomGuess"], y_tests["RandomGuess"] = y_random, y_test

    return results, y_preds, y_tests

#%% ===================== 5. 主流程 ======================
# run() 主流程，回傳 y_test/y_pred/price_test
'''
results_dict 儲存 accuracy
pred_dict 儲存 y_pred
ytest_dict 儲存 y_test
price_dict 儲存 price_test（即測試集價格序列）
'''

def run(year='2015', month='01', tickers=[], lagN=5):
    df_all = load_stock_data(year, month)
    if not tickers:
        tickers = df_all['Ticker'].unique()
    feature_cfgs = [
        {"name": "Price", "use_volume_lag": False, "use_ta": False, "use_more": False},
        {"name": "Price+Volume", "use_volume_lag": True, "use_ta": False, "use_more": False},
        {"name": "Price+Volume+TA", "use_volume_lag": True, "use_ta": True, "use_more": False},
        {"name": "Price+Volume+TA+more", "use_volume_lag": True, "use_ta": True, "use_more": True}
    ]

    results_dict = {cfg["name"]: pd.DataFrame(index=tickers) for cfg in feature_cfgs}
    pred_dict = {cfg["name"]: {ticker: {} for ticker in tickers} for cfg in feature_cfgs}
    ytest_dict = {cfg["name"]: {ticker: {} for ticker in tickers} for cfg in feature_cfgs}
    price_dict = {cfg["name"]: {ticker: None for ticker in tickers} for cfg in feature_cfgs}

    for ticker in tqdm(tickers, desc="遍歷股票"):
        df = df_all[df_all['Ticker'] == ticker][[
            'LastTradePrice', 'Volume', 'VolumeWeightPrice', 'TotalTrades', 'MinSpread', 'MaxSpread'
        ]].copy()
        df.index = df_all[df_all['Ticker'] == ticker]['datetime']
        for feat_cfg in feature_cfgs:
            name = feat_cfg["name"]
            try:
                df_feat, feats = make_features(df.copy(), feat_cfg, lagN)
                if len(df_feat) < 100:
                    continue
                X, y = df_feat[feats].values, df_feat['y'].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
                # 標準化
                mean, std = X_train.mean(axis=0), X_train.std(axis=0)
                X_train, X_test = (X_train-mean)/std, (X_test-mean)/std
                price_test = df_feat['LastTradePrice'][-X_test.shape[0]:].values
                price_dict[name][ticker] = price_test

                res, y_preds, y_tests = evaluate_models(X_train, y_train, X_test, y_test, epochs=10)
                for model, acc in res.items():
                    results_dict[name].loc[ticker, model] = acc
                    pred_dict[name][ticker][model] = y_preds[model]
                    ytest_dict[name][ticker][model] = y_tests[model]
            except Exception as e:
                print(f"{ticker} {name} 失敗: {e}")

    # 匯出
    with pd.ExcelWriter(f"accuracy_all_stocks_{year}{month}.xlsx") as writer:
        for name, table in results_dict.items():
            table.to_excel(writer, sheet_name=name.replace('+', '_')[:30])
    print(f"✅ 已存檔 accuracy_all_stocks_{year}{month}.xlsx")

    return results_dict, pred_dict, ytest_dict, price_dict, tickers

#%% ====================== 6. 回測函式 ======================
#   回測函式（All-in/All-out）
def backtest_all_in_all_out(y_pred, price_test, initial_capital=100_000):
    """
    y_pred: 預測序列（1/-1），可為任意長度
    price_test: 對應的實際價格序列
    initial_capital: 初始資金
    return: 最終資產、策略年化收益率、資產曲線
    """
    cash, shares = initial_capital, 0
    asset_curve = []
    for i in range(len(y_pred)):
        if y_pred[i] > 0 and cash > 0:
            shares = cash / price_test[i]
            cash = 0
        elif y_pred[i] < 0 and shares > 0:
            cash = shares * price_test[i]
            shares = 0
        asset_curve.append(cash + shares * price_test[i])
    final_value = cash + shares * price_test[-1]
    total_return = (final_value - initial_capital) / initial_capital
    return final_value, total_return, np.array(asset_curve)

def backtest_buy_and_hold(price_test, initial_capital=100_000):
    shares = initial_capital / price_test[0]
    final_value = shares * price_test[-1]
    total_return = (final_value - initial_capital) / initial_capital
    asset_curve = price_test * shares
    return final_value, total_return, asset_curve

def backtest_dca(price_test, initial_capital=100_000):
    N = len(price_test)
    invest_per_period = initial_capital / N
    total_shares = 0
    asset_curve = []
    for i in range(N):
        shares_bought = invest_per_period / price_test[i]
        total_shares += shares_bought
        # 當期資產價值
        current_value = total_shares * price_test[i] + invest_per_period * (N - i - 1)
        asset_curve.append(current_value)
    final_value = total_shares * price_test[-1]
    total_return = (final_value - initial_capital) / initial_capital
    return final_value, total_return, np.array(asset_curve)






#%%# ====================== 7. 主程式入口 ======================
## 主程式入口，遍歷多檔股票、多個月、多組特徵

if __name__ == "__main__":
    # 你可以自訂多檔股票、多個月、多組特徵
    years=   ['2015', '2016', '2017'] #, '2017'] #, '2016', '2015']            # or ['2015','2017']
    months=  ['01', '12']                 # or ['01','12']
    # tickers= [] # means select all #['AAL',  'AAPL', 'NVDA', 'COST', 'YHOO']   # or只取 ['AAPL']
    # get all NASDAQ 100 tickers
    # tickers = ['AAL',  'AAPL'] #, 'NVDA', 'COST', 'YHOO']
    tickers = [] # means select all

    feature_names = [
        #"Price", 
        "Price+Volume", 
        #"Price+Volume+TA", 
        "Price+Volume+TA+more"
    ]
    model_names = [
        "LinearRegression", 
        "RandomForest", 
        "XGBoost", 
        "LSTM", 
        "CNN", 
        "Transformer",
        "RandomGuess"
        ]

    # 統整所有回測結果
    summary_list = []

    for year in tqdm(years):
        for month in months:

            print(f"處理 {year}年{month}月的股票資料...")
            results_dict, pred_dict, ytest_dict, price_dict, _ = run(
                year=year, month=month, tickers=tickers
            )
            
            # 若原始 tickers 為空，則從結果中獲取 (全部股票)
            if not tickers:
                feature_name= feature_names[0]  # 假設只處理第一組特徵
                tickers = list(price_dict[feature_name].keys())
            print(f"共處理 {len(tickers)} 檔股票")

            for feature_name in feature_names:
                
                for ticker in tickers: #[0:10]: # 只處理前 10 檔股票， # 可根據需要調整

                    price_test = price_dict[feature_name].get(ticker, None)
                    if price_test is None or len(price_test) < 10:
                        continue
                    # Buy & Hold
                    final_bh, ret_bh, _ = backtest_buy_and_hold(price_test)
                    # DCA
                    final_dca, ret_dca, _ = backtest_dca(price_test)
                    for model_name in model_names:
                        y_pred = pred_dict[feature_name][ticker].get(model_name, None)
                        y_test = ytest_dict[feature_name][ticker].get(model_name, None)
                        if y_pred is None or y_test is None:
                            continue
                        try:
                            acc = (np.sign(y_pred) == np.sign(y_test)).mean()
                            final_val, total_ret, asset_curve = backtest_all_in_all_out(y_pred, price_test)
                            summary_list.append({
                                "Year": year,
                                "Month": month,
                                "Ticker": ticker,
                                "Feature": feature_name,
                                "Model": model_name,
                                "Accuracy": acc,
                                "AllinOutAsset": final_val,
                                "AllinOutReturn": total_ret,                            
                                "BuyHoldReturn": ret_bh,
                                #"BuyHoldAsset": final_bh,
                                "DCA_Return": ret_dca,
                                #"DCA_Asset": final_dca,
                                "SampleSize": len(y_pred),
                            })
                        except Exception as e:
                            print(f"{ticker}-{feature_name}-{model_name} 回測失敗: {e}")

            # 匯出結果 DataFrame
            summary_df = pd.DataFrame(summary_list)
            #print(summary_df)
            # 也可存成 Excel
            summary_df.to_excel(f"all_backtest_summary_{year}_{month}.xlsx", index=False)
            print(f"✅ 結果已存檔 all_backtest_summary_{year}_{month}.xlsx")




    # 匯出結果 DataFrame
    summary_df = pd.DataFrame(summary_list)
    print(summary_df)
    # 也可存成 Excel
    summary_df.to_excel(f"all_backtest_summary.xlsx", index=False)
    print("✅ 所有回測結果已存檔 all_backtest_summary.xlsx")

# %%
print(f'{summary_df= }')
# %%

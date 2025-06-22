'''
我想要從此網頁下載 .zip 檔，
https://algoseek-public.s3.amazonaws.com/nasdaq100-1min.zip

最後再目前目錄底下展開成 以  f"nasdaq100_monthly/nasdaq100_ {yyyymm}.h5" 檔案結構的資料集
其中 yyyy 代表 4位數的年份，範圍從 2015~2017
mm 代表 2位數月份，範圍從 01~12

nasdaq100-1min.zip 內含 .csv.gz ，以日為單位作為一個壓縮檔案，我要解壓縮後，合併成以月分作為單位存成 f"nasdaq100_ {yyyymm}.h5"
請寫一支 python 程式，從網路下載程式、解壓縮、讀 csv、合併成月份 csv、最後在一個資料夾之下，形成 3年共36個 .h5 檔案。 
'''
import os
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ---- 1. 下載檔案 ----
ZIP_URL = "https://algoseek-public.s3.amazonaws.com/nasdaq100-1min.zip"
ZIP_FILE = "nasdaq100-1min.zip"
EXTRACT_DIR = "nasdaq100-1min"

H5KEY = "minute_data"  # HDF5 檔案的 key 名稱

if not os.path.exists(ZIP_FILE):
    print("正在下載壓縮檔...")
    with requests.get(ZIP_URL, stream=True) as r:
        r.raise_for_status()
        with open(ZIP_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
else:
    print("已存在壓縮檔，略過下載。")

# ---- 2. 解壓縮 .zip ----
if not os.path.exists(EXTRACT_DIR):
    print("正在解壓縮 .zip 檔...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
else:
    print("已解壓縮 .zip，略過此步驟。")

# ---- 3. 遍歷所有 .csv.gz，按年月合併 ----
output_dir = Path("nasdaq100_monthly")
output_dir.mkdir(exist_ok=True)

# 準備 {yyyymm: [file_path, ...]}
files_by_yyyymm = {}

# 搜尋所有 .csv.gz 檔案
for path in Path(EXTRACT_DIR).rglob("*.csv.gz"):
    # 路徑格式： nasdaq100-1min/2015/20150102/AAPL.csv.gz
    # 取年、月
    parts = path.parts
    yyyy = parts[-3]
    yyyymmdd = parts[-2]
    yyyymm = yyyymmdd[:6]
    if yyyy not in ["2015", "2016", "2017"]:
        continue
    if int(yyyymm[4:6]) < 1 or int(yyyymm[4:6]) > 12:
        continue
    files_by_yyyymm.setdefault(yyyymm, []).append(path)

# ---- 4. 按月合併所有 csv.gz，儲存成 .h5 ----
for y in range(2015, 2018):
    for m in range(1, 13):
        yyyymm = f"{y}{m:02d}"
        if yyyymm not in files_by_yyyymm:
            print(f"缺少 {yyyymm} 資料，跳過")
            continue
        month_files = files_by_yyyymm[yyyymm]
        print(f"正在處理 {yyyymm}，共有 {len(month_files)} 個檔案...")

        # 讀進所有 csv.gz 並合併
        dfs = []
        for f in tqdm(month_files, desc=f"Reading {yyyymm}"):
            try:
                df = pd.read_csv(f, compression='gzip')
                # 可選：加上 ticker, yyyymmdd 欄位方便後續查詢
                ticker = f.name.split(".")[0]
                date = f.parts[-2]
                df['Ticker'] = ticker
                df['Date'] = date
                dfs.append(df)
            except Exception as e:
                print(f"讀取 {f} 發生錯誤: {e}")
        if dfs:
            month_df = pd.concat(dfs, ignore_index=True)
            out_path = output_dir / f"nasdaq100_{yyyymm}.h5"
            month_df.to_hdf(out_path, key= H5KEY, mode="w")
            print(f"存檔完成: {out_path}, 共 {len(month_df)} 筆")
        else:
            print(f"{yyyymm} 無有效資料")

print("全部完成！")

#%%
import pandas as pd
from pathlib import Path

h5_dir = Path("nasdaq100_monthly")
csv_dir = Path("csv")
csv_dir.mkdir(exist_ok=True)

for h5_file in h5_dir.glob("*.h5"):
    print(f"正在轉換 {h5_file.name} ...")
    df = pd.read_hdf(h5_file, key= H5KEY)
    csv_path = csv_dir / (h5_file.stem + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"已存成 {csv_path}")

print("全部完成！")
#%%
'''
你的需求是把3年內所有相同 ticker 的 minute bar 資料，合併成一個大的 csv，並放在 csv_by_ticker/{ticker_name}.csv。

兩種來源都可以（從 .h5 或從 .csv.gz），
但如果直接從 nasdaq100-1min 下的所有 .csv.gz 檔直接組合，效能更佳，也較省記憶體！

以下是推薦做法（從原始 nasdaq100-1min 目錄下所有 .csv.gz 檔整理），一次性低記憶體分批寫入，不需全部資料吃進 RAM！
'''
import pandas as pd
from pathlib import Path
from tqdm import tqdm

src_dir = Path("nasdaq100-1min")
out_dir = Path("csv_by_ticker")
out_dir.mkdir(exist_ok=True)

# 先找出所有 ticker 名稱
all_csvs = list(src_dir.rglob("*.csv.gz"))
ticker_set = set(f.name.split(".")[0] for f in all_csvs)

for ticker in tqdm(sorted(ticker_set)):
    # 找出所有屬於這個 ticker 的 csv.gz
    ticker_files = [f for f in all_csvs if f.name.startswith(f"{ticker}.")]
    out_file = out_dir / f"{ticker}.csv"
    is_first = True
    for f in sorted(ticker_files):  # 建議加 sorted 讓結果有序
        try:
            df = pd.read_csv(f, compression='gzip')
            # 加上日期欄
            date_str = f.parts[-2]
            df["Date"] = date_str
            # 第一次寫入用寫檔，之後 append
            df.to_csv(out_file, mode='w' if is_first else 'a', index=False, header=is_first)
            is_first = False
        except Exception as e:
            print(f"讀取 {f} 發生錯誤: {e}")
    print(f"已完成 {ticker}")

print("全部完成！")

# %%

#%% do some stats about the results
#   get results from f'accuracy_all_stocks_2015{month:02d}.xlsx'
import pandas as pd
from pathlib import Path

#data_path= Path('C:/_ryDatasets/accuracy_all_stocks')
#summary_df.to_excel(f"all_backtest_summary.xlsx", index=False)
#print("✅ 所有回測結果已存檔 all_backtest_summary.xlsx")

summary_df= pd.read_excel("all_backtest_summary.xlsx")
#summary_df=  pd.read_excel("C:/_ryDatasets/all_backtest_summary_201712_ryBackup.xlsx")

# 檢查資料
print(f'{summary_df= }')
# %%

# %%
# 總結統計 分別取 max, mean, min
#   按年、月、特徵、模型分組，計算最大值、平均值、最小值

#   這裡使用 max 作為示例
smryMax= summary_df.groupby(['Year', 'Month', 'Feature', 'Model']).agg({
    'Accuracy': 'max',
    'AllinOutAsset': 'max',
    'AllinOutReturn': 'max',
    'BuyHoldReturn': 'max',
    'DCA_Return': 'max',
    'SampleSize': 'max'
}).reset_index().sort_values(by=['AllinOutReturn', 'Accuracy', 'Year', 'Month', 'Feature', 'Model'], 
                             ascending=False)
smryMax
# %%
# correlation matrix
correlation_matrix= smryMax.corr(numeric_only= True)
print(correlation_matrix)

# %%
# correlation matrix df
correlation_matrix_df= summary_df.corr(numeric_only= True)
print(correlation_matrix_df)

# %%
# 以 min, max, mean 分組
smryMin= summary_df.groupby(['Year', 'Month', 'Feature', 'Model']).agg({
    'Accuracy': 'min',
    'AllinOutAsset': 'min',
    'AllinOutReturn': 'min',
    'BuyHoldReturn': 'min',
    'DCA_Return': 'min',
    'SampleSize': 'min'
}).reset_index().sort_values(by=['AllinOutReturn', 'Accuracy', 'Year', 'Month', 'Feature', 'Model'], 
                             ascending=False)
smryMin
# %%
# 以 mean 分組
smryMean= summary_df.groupby(['Year', 'Month', 'Feature', 'Model']).agg({
    'Accuracy': 'mean',
    'AllinOutAsset': 'mean',
    'AllinOutReturn': 'mean',
    'BuyHoldReturn': 'mean',
    'DCA_Return': 'mean',
    'SampleSize': 'mean'
}).reset_index().sort_values(by=['AllinOutReturn', 'Accuracy', 'Year', 'Month', 'Feature', 'Model'], 
                             ascending=False)
smryMean
# %%
# %%
# 以 max 分組
smryMax= summary_df.groupby(['Year', 'Month', 'Feature', 'Model']).agg({
    'Accuracy': 'max',
    'AllinOutAsset': 'max',
    'AllinOutReturn': 'max',
    'BuyHoldReturn': 'max',
    'DCA_Return': 'max',
    'SampleSize': 'max'
}).reset_index().sort_values(by=['AllinOutReturn', 'Accuracy', 'Year', 'Month', 'Feature', 'Model'], 
                             ascending=False)
smryMax

# %%
#%% 自動產出聚合摘要與圖表


import matplotlib
import matplotlib.pyplot as plt

# 設定中文字型，這裡以 Windows 微軟正黑體為例
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 ['Noto Sans CJK TC'], ['PingFang TC']
plt.rcParams['axes.unicode_minus'] = False  # 負號正常顯示
plt.rcParams['axes.grid'] = True  # 全局預設開啟格線

# 如果你在 Linux，可能要先安裝 Noto Sans CJK：
# sudo apt-get install fonts-noto-cjk

# 假設 summary_df 已載入
# 若還沒：summary_df = pd.read_excel("all_backtest_summary_201712_ryBackup.xlsx")

# 1. 模型聚合摘要（平均、標準差、最大、最小）
group_cols = ['Model']
agg_dict = {
    'Accuracy': ['mean', 'std', 'max', 'min'],
    'AllinOutReturn': ['mean', 'std', 'max', 'min'],
    'BuyHoldReturn': ['mean', 'std', 'max', 'min'],
    'DCA_Return': ['mean', 'std', 'max', 'min']
}
model_summary = summary_df.groupby(group_cols).agg(agg_dict)
model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns.values]
model_summary = model_summary.sort_values('AllinOutReturn_mean', ascending=False)
print(model_summary)
model_summary.to_excel('model_performance_summary.xlsx')

# 2. 箱型圖：不同模型的回報率分布
plt.figure(figsize=(14,6))
model_summary['AllinOutReturn_mean'].plot(kind='bar', color='skyblue')
plt.title('各模型平均區間報酬率')
plt.ylabel('Mean Return')
plt.xlabel('Model')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 3. 長條圖：不同模型的平均報酬率、準確率
plt.figure(figsize=(10,6))
model_summary['AllinOutReturn_mean'].plot(kind='bar', color='skyblue', label='AllinOutReturn')
model_summary['BuyHoldReturn_mean'].plot(kind='bar', color='orange', alpha=0.5, label='BuyHoldReturn')
plt.title('各模型平均區間報酬率（AllinOut/B&H）')
plt.ylabel('Mean Return')
plt.xlabel('Model')
plt.legend()
plt.savefig('barchart_mean_return_by_model.png')
plt.show()

plt.figure(figsize=(10,6))
model_summary['Accuracy_mean'].plot(kind='bar', color='seagreen')
plt.title('各模型平均預測準確率')
plt.ylabel('Mean Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.savefig('barchart_mean_accuracy_by_model.png')
plt.show()

# 4. 產出中英文摘要表格
model_summary_zh = model_summary.rename(columns={
    'Accuracy_mean': '平均準確率',
    'AllinOutReturn_mean': 'AllinOut平均報酬率',
    'BuyHoldReturn_mean': '買進持有平均報酬率',
    'DCA_Return_mean': '定期定額平均報酬率'
})
model_summary_zh.to_excel('model_performance_summary_zh.xlsx')

print('✅ 聚合摘要、圖表與中英文表格已產出')

# %%
# %%
def main():    
    # 這裡可以放入主程式邏輯
    # 例如：呼叫函數或其他模組
    print(f'已完成執行 {__file__}')

if __name__ == "__main__":
    main()

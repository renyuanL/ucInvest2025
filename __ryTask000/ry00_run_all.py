
import ry01_nasdaq100_下載_按月存放
import ry02_stock_prediction_gpu003
import ry03_do_some_stats

'''
This script is a wrapper to run the entire process of 
1. downloading NASDAQ-100 data,
2. predicting stock prices using a GPU
3. performing statistical analysis.
'''

def main():
    # Step 1: Download NASDAQ-100 data and save it monthly
    ry01_nasdaq100_下載_按月存放.main()

    # Step 2: Predict stock prices using GPU
    ry02_stock_prediction_gpu003.main()

    # Step 3: Perform statistical analysis on the predictions
    ry03_do_some_stats.main()

if __name__ == "__main__":
    main()
import yfinance as yf
import pandas as pd
import twstock
from tqdm import tqdm
import time
import warnings

# 忽略 pandas 的警告訊息，保持畫面乾淨
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_stock_list():
    """取得台股所有上市櫃代號"""
    print("正在取得台股清單...")
    tse = twstock.twse
    otc = twstock.tpex
    
    stock_dict = {} 
    
    # 上市
    for code, info in tse.items():
        if info.type == '股票':
            stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code}
            
    # 上櫃
    for code, info in otc.items():
        if info.type == '股票':
            stock_dict[f"{code}.TWO"] = {'name': info.name, 'code': code}
            
    print(f"共取得 {len(stock_dict)} 檔股票代號")
    return stock_dict

def process_batch(tickers_batch, stock_dict):
    """批次處理一批股票"""
    results = []
    try:
        # 下載資料
        data = yf.download(tickers_batch, period="15mo", progress=False, auto_adjust=True)
        
        # 檢查資料是否為空
        if data.empty:
            return []

        # 確保取得收盤價 (Close)
        # yfinance 批次下載時，通常會回傳一個 DataFrame，Columns 是股票代號
        try:
            df_close = data['Close']
        except KeyError:
            return []
            
        # 如果批次只有 1 檔，它會變成 Series，要轉回 DataFrame 以便統一處理
        if isinstance(df_close, pd.Series):
            df_close = df_close.to_frame(name=tickers_batch[0])

        # 計算均線 (針對整個 DataFrame 運算)
        ma200_df = df_close.rolling(window=200).mean()
        
        # 取得最後一天的價格和均線
        last_prices = df_close.iloc[-1]
        last_ma200 = ma200_df.iloc[-1]

        # 這裡改用「下載回來的 Columns」來當迴圈，確保數據對應正確
        # 避免因為下載失敗的股票導致順序錯亂
        for ticker in df_close.columns:
            try:
                # 取得該股票的數據
                price = last_prices[ticker]
                ma200 = last_ma200[ticker]
                
                # 檢查數據有效性
                if pd.isna(price) or pd.isna(ma200) or ma200 == 0:
                    continue

                bias = ((price - ma200) / ma200) * 100
                
                # 從字典找回中文名稱
                # 注意：yfinance 有時會把 .TWO 變成其他格式，這裡做個防呆
                stock_info = stock_dict.get(ticker)
                if not stock_info:
                    continue

                results.append({
                    '代號': stock_info['code'],
                    '名稱': stock_info['name'],
                    '收盤價': round(float(price), 2),
                    '200日均線': round(float(ma200), 2),
                    '與年線差距(%)': round(float(bias), 2)
                })
            except Exception:
                continue
                
    except Exception as e:
        pass
        
    return results

def main():
    # 1. 取得所有代號
    try:
        stock_dict = get_stock_list()
    except Exception as e:
        print(f"清單取得失敗: {e}")
        return

    all_tickers = list(stock_dict.keys())
    final_data = []
    
    # 批次大小設定 50 比較穩定
    BATCH_SIZE = 50
    
    print(f"開始精確抓取，共 {len(all_tickers)} 檔，每批 {BATCH_SIZE} 檔...")
    
    for i in tqdm(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[i : i + BATCH_SIZE]
        batch_results = process_batch(batch, stock_dict)
        final_data.extend(batch_results)
        time.sleep(0.5) # 稍微休息避免被擋

    if not final_data:
        print("沒有抓取到資料。")
        return

    # 3. 轉成 DataFrame 並排序
    df = pd.DataFrame(final_data)
    
    # 依據與年線差距的絕對值排序
    df['abs_bias'] = df['與年線差距(%)'].abs()
    df_sorted = df.sort_values(by='abs_bias')
    df_sorted = df_sorted.drop(columns=['abs_bias']) 

    # 4. 存檔
    filename = "台股200MA篩選表_修正版.csv"
    df_sorted.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n執行完成！共成功抓取 {len(df_sorted)} 檔股票。")
    print(f"檔案已儲存為: {filename}")
    # 驗證前幾筆資料，確保價格不同
    print("前 5 名最接近年線的股票 (驗證價格是否不同)：")
    print(df_sorted.head(5))

if __name__ == "__main__":
    main()
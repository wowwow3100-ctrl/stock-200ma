import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time

# --- 1. 網頁設定 ---
st.set_page_config(page_title="台股200MA戰法(PRO)", layout="wide")

# --- 2. 核心功能區 ---
@st.cache_data(ttl=3600)  # 設定快取，1小時內不用重新抓清單
def get_stock_list():
    """取得台股清單 (上市+上櫃)"""
    tse = twstock.twse
    otc = twstock.tpex
    stock_dict = {}
    
    # 上市
    for code, info in tse.items():
        if info.type == '股票':
            stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code, 'group': info.group}
    # 上櫃
    for code, info in otc.items():
        if info.type == '股票':
            stock_dict[f"{code}.TWO"] = {'name': info.name, 'code': code, 'group': info.group}
            
    return stock_dict

def process_batch(tickers_batch, stock_dict):
    """批次處理股票數據"""
    results = []
    try:
        # 下載數據 (只抓收盤價，速度最快)
        data = yf.download(tickers_batch, period="15mo", progress=False, auto_adjust=True)
        
        if data.empty:
            return []

        # 處理資料格式 (相容性處理)
        try:
            df_close = data['Close']
        except KeyError:
            return []
            
        if isinstance(df_close, pd.Series):
            df_close = df_close.to_frame(name=tickers_batch[0])

        # 計算 200 日均線
        ma200_df = df_close.rolling(window=200).mean()
        
        # 取得最新一天的數據
        last_prices = df_close.iloc[-1]
        last_ma200 = ma200_df.iloc[-1]

        for ticker in df_close.columns:
            try:
                price = last_prices[ticker]
                ma200 = last_ma200[ticker]
                
                # 過濾無效數據
                if pd.isna(price) or pd.isna(ma200) or ma200 == 0:
                    continue

                # 計算乖離率
                bias = ((price - ma200) / ma200) * 100
                
                stock_info = stock_dict.get(ticker)
                if not stock_info:
                    continue

                status = "🟢年線上" if price >= ma200 else "🔴年線下"

                results.append({
                    '代號': stock_info['code'],
                    '名稱': stock_info['name'],
                    '收盤價': round(float(price), 2),
                    '200MA': round(float(ma200), 2),
                    '乖離率(%)': round(float(bias), 2),
                    '位置': status,
                    'abs_bias': abs(bias) # 排序用隱藏欄位
                })
            except Exception:
                continue
    except Exception:
        pass
    
    return results

# --- 3. 介面顯示區 ---
st.title("📈 台股 200MA 戰法篩選器 (雲端版)")
st.markdown("此工具專門篩選 **股價回測年線 (200MA)** 附近的潛力股。")

# 側邊欄控制
with st.sidebar:
    st.header("⚙️ 篩選條件")
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 3.0, step=0.5)
    st.caption("建議值：3% 以內代表剛好黏在年線附近。")
    
    run_btn = st.button("🚀 開始掃描", type="primary")

# 主畫面邏輯
if run_btn:
    st.divider()
    status_text = st.empty()
    progress_bar = st.progress(0, text="正在準備資料庫...")
    
    try:
        # 1. 取得清單
        stock_dict = get_stock_list()
        all_tickers = list(stock_dict.keys())
        
        status_text.info(f"鎖定全台 {len(all_tickers)} 檔股票，開始批次運算...")
        
        # 2. 批次執行
        BATCH_SIZE = 50
        total_batches = (len(all_tickers) // BATCH_SIZE) + 1
        final_data = []

        for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
            batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
            
            # 執行運算
            batch_results = process_batch(batch, stock_dict)
            final_data.extend(batch_results)
            
            # 更新進度條
            current_progress = (i + 1) / total_batches
            progress_bar.progress(current_progress, text=f"掃描進度：{int(current_progress*100)}%")
            
            # 稍微休息避免被擋
            time.sleep(0.1)
        
        progress_bar.empty() # 跑完隱藏進度條
        
        if final_data:
            # 3. 資料整理與顯示
            df = pd.DataFrame(final_data)
            
            # 篩選使用者設定的乖離率
            df = df[df['abs_bias'] <= bias_threshold]
            df = df.sort_values(by='abs_bias')
            
            status_text.success(f"✅ 掃描完成！共有 {len(df)} 檔股票符合條件。")

            # 分頁顯示
            tab1, tab2 = st.tabs(["🔥 站上年線 (多方)", "🧊 跌破年線 (觀察)"])
            
            with tab1:
                df_up = df[df['位置'] == "🟢年線上"].drop(columns=['位置', 'abs_bias'])
                st.dataframe(df_up, use_container_width=True, hide_index=True)
                
            with tab2:
                df_down = df[df['位置'] == "🔴年線下"].drop(columns=['位置', 'abs_bias'])
                st.dataframe(df_down, use_container_width=True, hide_index=True)
                
        else:
            status_text.warning("沒有抓取到符合條件的資料。")

    except Exception as e:
        st.error(f"發生未預期的錯誤: {e}")

import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime

# --- 1. 網頁設定 (版本號 +1) ---
VER = "ver1.0"
st.set_page_config(page_title=f"旺來戰法過濾器({VER})", layout="wide")

# --- 2. 核心功能區 ---
@st.cache_data(ttl=3600)
def get_stock_list():
    """取得台股清單 (排除金融/ETF)"""
    tse = twstock.twse
    otc = twstock.tpex
    stock_dict = {}
    
    exclude_industries = ['金融保險業', '存託憑證']

    for code, info in tse.items():
        if info.type == '股票':
            if info.group not in exclude_industries:
                stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code, 'group': info.group}
                
    for code, info in otc.items():
        if info.type == '股票':
            if info.group not in exclude_industries:
                stock_dict[f"{code}.TWO"] = {'name': info.name, 'code': code, 'group': info.group}
            
    return stock_dict

def calculate_kd_values(df, n=9):
    """計算並回傳最後一天的 K, D 值"""
    try:
        low_min = df['Low'].rolling(window=n).min()
        high_max = df['High'].rolling(window=n).max()
        
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        
        k, d = 50, 50
        # 簡單計算最後幾天的 KD 即可，不用算全部歷史，加速運算
        # 但為了準確，還是建議跑一個小迴圈
        k_list, d_list = [], []
        
        for r in rsv:
            k = (2/3) * k + (1/3) * r
            d = (2/3) * d + (1/3) * k
            k_list.append(k)
            d_list.append(d)
            
        return k_list[-1], d_list[-1]
    except:
        return 50, 50

def fetch_all_data(stock_dict, progress_bar, status_text):
    """【廚房】一次性下載並計算所有原始數據"""
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 30
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []

    # 批次下載
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        
        try:
            # 下載數據
            data = yf.download(batch, period="1y", progress=False, auto_adjust=False)
            
            if not data.empty:
                # 處理多層索引
                try:
                    df_c = data['Close']
                    df_h = data['High']
                    df_l = data['Low']
                    df_v = data['Volume']
                except KeyError:
                    continue

                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])

                # 計算指標
                ma200_series = df_c.rolling(window=200).mean().iloc[-1]
                last_price_series = df_c.iloc[-1]
                last_vol_series = df_v.iloc[-1]
                prev_vol_series = df_v.iloc[-2]

                for ticker in df_c.columns:
                    try:
                        price = last_price_series[ticker]
                        ma200 = ma200_series[ticker]
                        vol = last_vol_series[ticker]
                        prev_vol = prev_vol_series[ticker]
                        
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0:
                            continue

                        # 計算 KD (針對單檔)
                        # 為了效能，這裡只對有基本資料的股票算
                        stock_df = pd.DataFrame({
                            'Close': df_c[ticker], 'High': df_h[ticker], 'Low': df_l[ticker]
                        }).dropna()
                        
                        k_val, d_val = 0, 0
                        if len(stock_df) >= 9:
                            k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue

                        # 存入原始資料庫 (不進行篩選，全部存下來)
                        raw_data_list.append({
                            '代號': stock_info['code'],
                            '名稱': stock_info['name'],
                            '收盤價': float(price),
                            '200MA': float(ma200),
                            '乖離率(%)': float(bias),
                            'abs_bias': abs(float(bias)), # 用於排序
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': float(k_val),
                            'D值': float(d_val),
                            '位置': "🟢年線上" if price >= ma200 else "🔴年線下"
                        })
                    except:
                        continue
        except:
            pass

        # 更新進度
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"資料下載中...({int(current_progress*100)}%)")
        time.sleep(0.05)
    
    return pd.DataFrame(raw_data_list)

# --- 3. 介面顯示區 ---
st.title(f"🍍 {VER} 旺來戰法過濾器")
st.markdown("---")

# 初始化 Session State (資料保溫箱)
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = None
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None

# 側邊欄：控制面板
with st.sidebar:
    st.header("1. 資料庫管理")
    
    # 更新按鈕
    if st.button("🔄 更新股價資料 (開市請按我)", type="primary"):
        stock_dict = get_stock_list()
        status_text = st.empty()
        progress_bar = st.progress(0, text="準備下載...")
        
        # 呼叫廚房煮菜
        df = fetch_all_data(stock_dict, progress_bar, status_text)
        
        # 存入保溫箱
        st.session_state['master_df'] = df
        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        progress_bar.empty()
        st.success(f"更新完成！共 {len(df)} 檔資料")
        
    if st.session_state['last_update']:
        st.caption(f"最後更新：{st.session_state['last_update']}")
    
    st.divider()
    
    st.header("2. 即時篩選器 (免等待)")
    
    # 這裡的調整會「即時」反應，不用重新下載
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0,

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
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    
    st.subheader("進階條件")
    filter_kd = st.checkbox("KD 黃金交叉 (K > D)", value=False)
    filter_vol_double = st.checkbox("爆量 (今日 > 昨日x2)", value=False)
    filter_ma_up = st.checkbox("只看站上年線 (多方)", value=False)

# 主畫面：展示區
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    # --- 執行篩選邏輯 (在本地端快速運算) ---
    
    # 1. 乖離率篩選
    df = df[df['abs_bias'] <= bias_threshold]
    
    # 2. 成交量篩選 (資料庫存的是股數，這裡輸入是張數)
    df = df[df['成交量'] >= (min_vol_input * 1000)]
    
    # 3. KD 篩選
    if filter_kd:
        # K > D 且 K < 80 (追高風險) 且 K > 20 (低檔鈍化) - 可依需求調整，這裡先只做 K>D
        df = df[df['K值'] > df['D值']]
        
    # 4. 爆量篩選
    if filter_vol_double:
        df = df[df['成交量'] > (df['昨日成交量'] * 2)]
        
    # 5. 只看年線上
    if filter_ma_up:
        df = df[df['位置'] == "🟢年線上"]

    # --- 顯示結果 ---
    st.info(f"根據目前條件，共篩選出 {len(df)} 檔股票")
    
    # 整理顯示欄位
    df['成交量(張)'] = (df['成交量'] / 1000).astype(int)
    df['KD值'] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
    
    display_cols = ['代號', '名稱', '收盤價', '成交量(張)', '乖離率(%)', '位置', 'KD值']
    
    # 排序
    df = df.sort_values(by='abs_bias')
    
    # 分頁顯示
    tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 統計數據"])
    
    with tab1:
        # 依照位置上色
        def highlight_pos(val):
            color = '#d1ffbd' if val == "🟢年線上" else '#ffcccc'
            return f'background-color: {color}'
            
        st.dataframe(
            df[display_cols].style.map(highlight_pos, subset=['位置']),
            use_container_width=True,
            hide_index=True
        )
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("站上年線數量", len(df[df['位置']=="🟢年線上"]))
        with col2:
            st.metric("跌破年線數量", len(df[df['位置']=="🔴年線下"]))

else:
    # 如果還沒下載過資料
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 更新股價資料」** 按鈕開始下載數據！")
    st.markdown("""
    ### 📖 旺來戰法使用說明
    1. **第一次使用**：請務必按左上角的更新按鈕，這會花一點時間把全台股票抓下來。
    2. **篩選免等待**：資料抓好後，調整下方的滑桿或勾選框，右邊表格會 **「瞬間」** 更新，不用重跑！
    3. **省時省力**：除非你需要最新的盤中股價，否則下載一次可以用一整天。
    """)

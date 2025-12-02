import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime
import plotly.graph_objects as go # 引入繪圖套件

# --- 1. 網頁設定 ---
VER = "ver1.1"
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
        for r in rsv:
            k = (2/3) * k + (1/3) * r
            d = (2/3) * d + (1/3) * k
        return k, d
    except:
        return 50, 50

def fetch_all_data(stock_dict, progress_bar, status_text):
    """【廚房】一次性下載所有數據"""
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 30
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", progress=False, auto_adjust=False)
            if not data.empty:
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
                        
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue

                        # 簡易 KD 計算
                        stock_df = pd.DataFrame({'Close': df_c[ticker], 'High': df_h[ticker], 'Low': df_l[ticker]}).dropna()
                        k_val, d_val = 0, 0
                        if len(stock_df) >= 9:
                            k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue

                        raw_data_list.append({
                            '代號': stock_info['code'],
                            '名稱': stock_info['name'],
                            '完整代號': ticker, # 存這個方便畫圖用
                            '收盤價': float(price),
                            '200MA': float(ma200),
                            '乖離率(%)': float(bias),
                            'abs_bias': abs(float(bias)),
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': float(k_val),
                            'D值': float(d_val),
                            '位置': "🟢年線上" if price >= ma200 else "🔴年線下"
                        })
                    except: continue
        except: pass
        
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"資料下載中...({int(current_progress*100)}%)")
        time.sleep(0.05)
    
    return pd.DataFrame(raw_data_list)

def plot_stock_chart(ticker, name):
    """繪製個股 K 線 + 200MA 圖表"""
    try:
        # 即時抓取該股歷史資料
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=False)
        if df.empty:
            st.error("無法取得數據")
            return

        # 計算 200MA
        df['200MA'] = df['Close'].rolling(window=200).mean()
        
        # 建立互動圖表
        fig = go.Figure()

        # 1. K線圖
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='K線'
        ))

        # 2. 200MA 線
        fig.add_trace(go.Scatter(
            x=df.index, y=df['200MA'],
            line=dict(color='orange', width=2),
            name='200MA (年線)'
        ))

        fig.update_layout(
            title=f"{name} ({ticker}) - 年線攻防戰",
            yaxis_title='股價',
            xaxis_rangeslider_visible=False, # 隱藏下方滑桿
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"繪圖失敗: {e}")

# --- 3. 介面顯示區 ---
st.title(f"🍍 {VER} 旺來戰法過濾器")
st.markdown("---")

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = None
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None

with st.sidebar:
    st.header("1. 資料庫管理")
    if st.button("🔄 更新股價資料 (開市請按我)", type="primary"):
        stock_dict = get_stock_list()
        status_text = st.empty()
        progress_bar = st.progress(0, text="準備下載...")
        df = fetch_all_data(stock_dict, progress_bar, status_text)
        st.session_state['master_df'] = df
        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress_bar.empty()
        st.success(f"更新完成！共 {len(df)} 檔資料")
        
    if st.session_state['last_update']:
        st.caption(f"最後更新：{st.session_state['last_update']}")
    
    st.divider()
    st.header("2. 即時篩選器")
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    st.subheader("進階條件")
    filter_kd = st.checkbox("KD 黃金交叉 (K > D)", value=False)
    filter_vol_double = st.checkbox("爆量 (今日 > 昨日x2)", value=False)
    filter_ma_up = st.checkbox("只看站上年線 (多方)", value=False)
    
    st.divider()
    # --- 版本紀錄按鈕 ---
    with st.expander("📅 版本開發紀錄"):
        st.markdown("""
        **Ver 1.1 (Chart Upgrade)**
        - 新增個股 K 線圖與 200MA 視覺化分析。
        - 側邊欄新增版本紀錄按鈕。
        
        **Ver 1.0 (Architecture)**
        - 資料下載與篩選分離，篩選免等待。
        - 排除金融股與 ETF。
        
        **Ver 0.9 (Calibration)**
        - 校正 200MA 數值 (使用原始收盤價)。
        
        **Ver 0.5 (Cloud)**
        - 部署至 Streamlit Cloud。
        """)

# 主畫面
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    # 篩選邏輯
    df = df[df['abs_bias'] <= bias_threshold]
    df = df[df['成交量'] >= (min_vol_input * 1000)]
    if filter_kd: df = df[df['K值'] > df['D值']]
    if filter_vol_double: df = df[df['成交量'] > (df['昨日成交量'] * 2)]
    if filter_ma_up: df = df[df['位置'] == "🟢年線上"]

    st.info(f"根據目前條件，共篩選出 {len(df)} 檔股票")
    
    # 整理顯示
    df['成交量(張)'] = (df['成交量'] / 1000).astype(int)
    df['KD值'] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
    df['選股標籤'] = df['代號'] + " " + df['名稱'] # 製作下拉選單用的標籤
    
    display_cols = ['代號', '名稱', '收盤價', '成交量(張)', '乖離率(%)', '位置', 'KD值']
    df = df.sort_values(by='abs_bias')
    
    tab1, tab2, tab3 = st.tabs(["📋 篩選結果列表", "📊 K線技術分析", "📈 統計數據"])
    
    with tab1:
        def highlight_pos(val):
            color = '#d1ffbd' if val == "🟢年線上" else '#ffcccc'
            return f'background-color: {color}'
        st.dataframe(df[display_cols].style.map(highlight_pos, subset=['位置']), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 🔍 個股詳細技術分析")
        if len(df) > 0:
            # 製作選單，預設選第一檔
            selected_stock_label = st.selectbox("請選擇一檔股票查看圖表：", df['選股標籤'].tolist())
            
            # 找出對應的完整代號 (xxx.TW)
            selected_row = df[df['選股標籤'] == selected_stock_label].iloc[0]
            target_ticker = selected_row['完整代號']
            target_name = selected_row['名稱']
            
            # 畫圖
            plot_stock_chart(target_ticker, target_name)
            
            # 顯示該股數據
            col1, col2, col3 = st.columns(3)
            col1.metric("目前股價", selected_row['收盤價'])
            col2.metric("200日均線", selected_row['200MA'], delta=f"{selected_row['乖離率(%)']}%")
            col3.metric("KD指標", selected_row['KD值'])
        else:
            st.warning("目前沒有篩選出股票，無法畫圖。")

    with tab3:
        col1, col2 = st.columns(2)
        with col1: st.metric("站上年線數量", len(df[df['位置']=="🟢年線上"]))
        with col2: st.metric("跌破年線數量", len(df[df['位置']=="🔴年線下"]))

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 更新股價資料」** 按鈕開始下載數據！")

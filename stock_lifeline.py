import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import os
import traceback 

# --- 1. 網頁設定 ---
VER = "ver5.4_Industry"
st.set_page_config(page_title=f"🍍 旺來-台股生命線({VER})", layout="wide")

# --- 2. 核心功能區 ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list():
    """
    從 twstock 抓取股票清單，並新增 'group' (產業類別) 資訊。
    """
    try:
        tse = twstock.twse
        otc = twstock.tpex
        stock_dict = {}
        exclude = ['金融保險業', '存託憑證']
        
        # 處理上市股票
        for code, info in tse.items():
            if info.type == '股票' and info.group not in exclude:
                # ★★★ 新增：儲存 group (產業) 資訊 ★★★
                stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code, 'group': info.group}
        
        # 處理上櫃股票
        for code, info in otc.items():
            if info.type == '股票' and info.group not in exclude:
                stock_dict[f"{code}.TWO"] = {'name': info.name, 'code': code, 'group': info.group}
                
        return stock_dict
    except: return {}

def calculate_kd(df, n=9):
    try:
        low_min = df['Low'].rolling(n).min()
        high_max = df['High'].rolling(n).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        k, d = 50, 50
        for r in rsv:
            k = (2/3) * k + (1/3) * r
            d = (2/3) * d + (1/3) * k
        return k, d
    except: return 50, 50

# --- 3. 策略擂台運算核心 ---
def run_optimization(stock_dict, progress_bar):
    raw_signals = [] 
    all_tickers = list(stock_dict.keys())
    BATCH = 50 
    total_batches = (len(all_tickers) // BATCH) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH)):
        batch = all_tickers[batch_idx : batch_idx + BATCH]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex): pass 
            
            if not data.empty:
                try:
                    df_c, df_v = data['Close'], data['Volume']
                    df_l, df_h = data['Low'], data['High']
                except: continue
                
                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])
                    df_l, df_h = df_l.to_frame(name=batch[0]), df_h.to_frame(name=batch[0])

                ma20 = df_c.rolling(20).mean()
                ma60 = df_c.rolling(60).mean()
                ma200 = df_c.rolling(200).mean()
                scan_idx = df_c.index[-250:-25]
                
                for ticker in df_c.columns:
                    try:
                        c, v = df_c[ticker], df_v[ticker]
                        l, h = df_l[ticker], df_h[ticker]
                        m200, m20, m60 = ma200[ticker], ma20[ticker], ma60[ticker]
                        
                        if c.isna().sum() > 100: continue

                        for date in scan_idx:
                            if pd.isna(m200[date]): continue
                            idx = c.index.get_loc(date)
                            if idx < 60: continue 

                            cp, lp = float(c.iloc[idx]), float(l.iloc[idx])
                            vol, p_vol = float(v.iloc[idx]), float(v.iloc[idx-1])
                            m200v, m20v, m60v = float(m200.iloc[idx]), float(m20.iloc[idx]), float(m60.iloc[idx])
                            
                            if m200v == 0 or p_vol == 0: continue

                            cond_near = (lp <= m200v * 1.03) and (lp >= m200v * 0.90)
                            cond_up = (cp > m200v)
                            basic = cond_near and cond_up
                            trend_up = (m200v > float(m200.iloc[idx-20]))
                            vol_dbl = (vol > p_vol * 1.5)
                            crown = (cp > m20v) and (m20v > m60v) and (m60v > m200v) and trend_up

                            treasure = False
                            if idx >= 7:
                                rc, rm = c.iloc[idx-7:idx+1], m200.iloc[idx-7:idx+1]
                                if rc.iloc[-1] > rm.iloc[-1] and (rc.iloc[:-1] < rm.iloc[:-1]).any():
                                    treasure = True

                            if not basic and not treasure and not crown: continue
                                
                            if idx + 20 < len(c):
                                ret_s = (float(c.iloc[idx+20]) - cp) / cp * 100
                                win_s = ret_s > 0

                                exit_d = float(c.iloc[idx+20])
                                for fi in range(1, 21):
                                    fidx = idx + fi
                                    if fidx >= len(c): break
                                    if float(h.iloc[fidx]) >= cp * 1.10: 
                                        exit_d = cp * 1.10
                                        break
                                    if float(c.iloc[fidx]) < float(m200.iloc[fidx]) * 0.99: 
                                        exit_d = float(c.iloc[fidx])
                                        break
                                
                                ret_d = (exit_d - cp) / cp * 100
                                win_d = ret_d > 0
                                
                                raw_signals.append({
                                    'P_Static': ret_s, 'W_Static': win_s,
                                    'P_Dynamic': ret_d, 'W_Dynamic': win_d,
                                    'Trend': trend_up, 'Vol': vol_dbl, 'Treasure': treasure,
                                    'Crown': crown, 'Basic': basic
                                })
                    except: continue
        except: pass
        progress_bar.progress((i+1)/total_batches, text="策略掃描中...")
        
    return pd.DataFrame(raw_signals)

# --- 4. 單一策略詳細回測 ---
def run_backtest(stock_dict, pbar, trend, treasure, vol, crown):
    results = []
    tickers = list(stock_dict.keys())
    BATCH = 50
    for i, b_idx in enumerate(range(0, len(tickers), BATCH)):
        batch = tickers[b_idx:b_idx+BATCH]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex): pass
            if not data.empty:
                try:
                    df_c, df_v = data['Close'], data['Volume']
                    df_l, df_h = data['Low'], data['High']
                except: continue
                if isinstance(df_c, pd.Series): 
                    df_c = df_c.to_frame(name=batch[0])
                    df_v, df_l, df_h = df_v.to_frame(name=batch[0]), df_l.to_frame(name=batch[0]), df_h.to_frame(name=batch[0])
                
                ma20, ma60, ma200 = df_c.rolling(20).mean(), df_c.rolling(60).mean(), df_c.rolling(200).mean()
                scan = df_c.index[-250:-25]

                for tk in df_c.columns:
                    try:
                        c, v, l, h = df_c[tk], df_v[tk], df_l[tk], df_h[tk]
                        m200, m20, m60 = ma200[tk], ma20[tk], ma60[tk]
                        
                        # ★★★ 新增：取得產業資訊 ★★★
                        stock_info = stock_dict.get(tk, {})
                        name = stock_info.get('name', tk)
                        industry = stock_info.get('group', '其他')

                        for date in scan:
                            if pd.isna(m200[date]): continue
                            idx = c.index.get_loc(date)
                            if idx < 60: continue
                            
                            cp, lp, vol_val = float(c.iloc[idx]), float(l.iloc[idx]), float(v.iloc[idx])
                            m200v = float(m200.iloc[idx])
                            if m200v==0: continue

                            match = False
                            if crown:
                                is_trend = m200v > float(m200.iloc[idx-20])
                                is_order = (cp > float(m20.iloc[idx])) and (float(m20.iloc[idx]) > float(m60.iloc[idx])) and (float(m60.iloc[idx]) > m200v)
                                if is_trend and is_order: match = True
                            else:
                                if trend and m200v <= float(m200.iloc[idx-20]): continue
                                if vol and vol_val <= float(v.iloc[idx-1])*1.5: continue
                                if treasure:
                                    if idx>=7:
                                        rc, rm = c.iloc[idx-7:idx+1], m200.iloc[idx-7:idx+1]
                                        if rc.iloc[-1]>rm.iloc[-1] and (rc.iloc[:-1]<rm.iloc[:-1]).any(): match = True
                                else:
                                    if lp <= m200v*1.03 and lp >= m200v*0.90 and cp > m200v: match = True
                            
                            if match and idx+20 < len(c):
                                ep, status = float(c.iloc[idx+20]), "持有20天"
                                if crown:
                                    for fi in range(1, 21):
                                        fidx = idx+fi
                                        if fidx>=len(c): break
                                        if float(h.iloc[fidx]) >= cp*1.1:
                                            ep, status = cp*1.1, "🎯停利"
                                            break
                                        if float(c.iloc[fidx]) < float(m200.iloc[fidx])*0.99:
                                            ep, status = float(c.iloc[fidx]), "🛡️停損"
                                            break
                                ret = (ep - cp)/cp*100
                                # ★★★ 修改：將欄位名稱中文化，並加入產業 ★★★
                                results.append({
                                    '日期': date.strftime('%Y-%m-%d'), 
                                    '代號': tk.replace('.TW','').replace('.TWO',''), 
                                    '名稱': name, 
                                    '產業': industry,
                                    '觸發價': round(cp, 2), 
                                    '報酬%': round(ret, 2), 
                                    '結果': status
                                })
                    except: continue
        except: pass
        pbar.progress((i+1)/((len(tickers)//BATCH)+1), text="回測運算中...")
    return pd.DataFrame(results)

# --- 5. 即時資料抓取 ---
def fetch_data(stock_dict, pbar):
    if not stock_dict: return pd.DataFrame()
    tickers = list(stock_dict.keys())
    BATCH = 30
    res = []
    for i, b_idx in enumerate(range(0, len(tickers), BATCH)):
        batch = tickers[b_idx:b_idx+BATCH]
        try:
            data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex): pass
            if not data.empty:
                try: df_c, df_h, df_l, df_v = data['Close'], data['High'], data['Low'], data['Volume']
                except: continue
                if isinstance(df_c, pd.Series): 
                    df_c = df_c.to_frame(name=batch[0])
                    df_h, df_l, df_v = df_h.to_frame(name=batch[0]), df_l.to_frame(name=batch[0]), df_v.to_frame(name=batch[0])

                m200, m20, m60 = df_c.rolling(200).mean(), df_c.rolling(20).mean(), df_c.rolling(60).mean()
                
                for tk in df_c.columns:
                    try:
                        p = float(df_c[tk].iloc[-1])
                        m200v = float(m200[tk].iloc[-1])
                        if pd.isna(p) or m200v==0: continue
                        
                        m20v, m60v = float(m20[tk].iloc[-1]), float(m60[tk].iloc[-1])
                        
                        crown = (p > m20v) and (m20v > m60v) and (m60v > m200v) and (m200v > float(m200[tk].iloc[-21]))
                        treasure = False
                        rc, rm = df_c[tk].iloc[-8:], m200[tk].iloc[-8:]
                        if len(rc)>=8 and rc.iloc[-1]>rm.iloc[-1] and (rc.iloc[:-1]<rm.iloc[:-1]).any(): treasure = True
                        
                        sdf = pd.DataFrame({'Close':df_c[tk], 'High':df_h[tk], 'Low':df_l[tk]}).dropna()
                        k, d = calculate_kd(sdf) if len(sdf)>=9 else (0,0)
                        
                        bias = (p - m200v)/m200v * 100
                        
                        # ★★★ 新增：取得產業資訊 ★★★
                        info = stock_dict.get(tk, {})
                        industry = info.get('group', '其他')
                        
                        res.append({
                            '代號': info.get('code',''), 
                            '名稱': info.get('name',''), 
                            '產業': industry, # 新增這個欄位
                            '完整代號': tk,
                            '收盤': round(p,2), 
                            '生命線': round(m200v,2), 
                            '乖離': round(bias,2), 
                            'abs_bias': abs(bias),
                            '量': int(df_v[tk].iloc[-1]), 
                            '昨量': int(df_v[tk].iloc[-2]),
                            '位置': "線上" if p>=m200v else "線下",
                            '浴火': treasure, '皇冠': crown, 'KD': f"K{int(k)}D{int(d)}"
                        })
                    except: continue
        except: pass
        pbar.progress((i+1)/((len(tickers)//BATCH)+1), text="即時股價更新中...")
        time.sleep(0.02)
    return pd.DataFrame(res)

def plot_chart(ticker, name):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if df.empty: return
        df['MA200'], df['MA60'], df['MA20'] = df['Close'].rolling(200).mean(), df['Close'].rolling(60).mean(), df['Close'].rolling(20).mean()
        pdf = df.tail(120).copy()
        pdf['Date'] = pdf.index.strftime('%Y-%m-%d')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pdf['Date'], y=pdf['Close'], name='收盤', line=dict(color='#00CC96')))
        fig.add_trace(go.Scatter(x=pdf['Date'], y=pdf['MA20'], name='月線', line=dict(color='#AB63FA', width=1)))
        fig.add_trace(go.Scatter(x=pdf['Date'], y=pdf['MA60'], name='季線', line=dict(color='#19D3F3', width=1)))
        fig.add_trace(go.Scatter(x=pdf['Date'], y=pdf['MA200'], name='生命線', line=dict(color='#FFA15A', width=3)))
        fig.update_layout(title=f"{name} ({ticker})", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    except: st.error("繪圖失敗")

# --- 6. 主應用程式介面 ---
def main_app():
    st.title(f"🍍 {VER} 旺來-台股生命線")
    st.markdown("---")

    if 'mdf' not in st.session_state: st.session_state['mdf'] = None
    if 'opt' not in st.session_state: st.session_state['opt'] = None
    if 'bt' not in st.session_state: st.session_state['bt'] = None

    with st.sidebar:
        st.header("設定")
        if st.button("🚨 重置"): st.cache_data.clear(); st.session_state.clear(); st.rerun()
        
        st.info("💡 歡迎使用！祝您操作順利，天天漲停！")
        
        if st.button("🔄 更新股價", type="primary"):
            sdict = get_stock_list()
            if sdict:
                pb = st.progress(0, "下載中...")
                st.session_state['mdf'] = fetch_data(sdict, pb)
                pb.empty()
                st.success("完成")
        
        st.divider()
        bias = st.slider("乖離率", 0.5, 20.0, 5.0)
        vol_min = st.number_input("最小量", 1000, step=100)
        
        st.subheader("篩選條件")
        f_up = st.checkbox("📈 生命線向上")
        f_tr = st.checkbox("🔥 浴火重生")
        f_cr = st.checkbox("👑 皇冠特選 (多頭+動態)")
        f_vo = st.checkbox("出量 (>1.5倍)")
        
        st.divider()
        st.subheader("分析工具")
        if st.button("🏆 策略擂台 (分析勝率)"):
            sdict = get_stock_list()
            pb = st.progress(0)
            st.session_state['opt'] = run_optimization(sdict, pb)
            pb.empty()
        
        if st.button("🧪 單一回測 (歷史交易)"):
            sdict = get_stock_list()
            pb = st.progress(0)
            st.session_state['bt'] = run_backtest(sdict, pb, f_up, f_tr, f_vo, f_cr)
            pb.empty()

    # --- 顯示區塊 ---
    
    # A. 策略擂台結果
    if st.session_state['opt'] is not None:
        df = st.session_state['opt']
        st.subheader("🏆 擂台結果 (持有20天 vs 動態出場)")
        if not df.empty:
            s_list = []
            strats = {
                "1. 裸測": df[df['Basic']],
                "2. 順勢": df[df['Basic'] & df['Trend']],
                "3. 爆量": df[df['Basic'] & df['Vol']],
                "4. 浴火": df[df['Treasure']],
                "7. 👑 皇冠(動態)": df[df['Crown']]
            }
            for n, d in strats.items():
                if len(d)>0:
                    is_dyn = "皇冠" in n
                    w = len(d[d['W_Dynamic']]) if is_dyn else len(d[d['W_Static']])
                    p = d['P_Dynamic'].mean() if is_dyn else d['P_Static'].mean()
                    s_list.append({'策略':n, '次數':len(d), '勝率%': (w/len(d))*100, '報酬%': p})
            
            res = pd.DataFrame(s_list).sort_values('勝率%', ascending=False)
            st.dataframe(res.style.background_gradient(subset=['勝率%', '報酬%'], cmap='RdYlGn'), use_container_width=True)

    # B. 單一策略回測結果
    if st.session_state['bt'] is not None:
        df = st.session_state['bt']
        st.subheader("🧪 回測報告")
        if not df.empty:
            win = len(df[df['報酬%']>0])
            st.metric("勝率", f"{int(win/len(df)*100)}%", f"均報 {round(df['報酬%'].mean(),2)}%")
            # 調整欄位順序，讓產業排前面一點
            cols = ['日期', '代號', '名稱', '產業', '觸發價', '報酬%', '結果']
            st.dataframe(df[cols].style.map(lambda v: f'color: {"red" if v>0 else "green"}', subset=['報酬%']), use_container_width=True)
        else: st.warning("無資料")

    # C. 日常篩選列表
    if st.session_state['mdf'] is not None:
        df = st.session_state['mdf'].copy()
        df = df[(df['abs_bias']<=bias) & (df['量']>=vol_min)]
        if f_up: df = df[df['生命線'] < df['收盤']]
        if f_tr: df = df[df['浴火']]
        if f_cr: df = df[df['皇冠']]
        if f_vo: df = df[df['量'] > df['昨量']*1.5]
        
        st.success(f"篩出 {len(df)} 檔")
        c1, c2 = st.columns([1.5, 1])
        with c1: 
            # 調整即時列表的顯示欄位
            display_cols = ['代號', '名稱', '產業', '收盤', '生命線', '乖離', '量', '浴火', '皇冠', 'KD']
            st.dataframe(df[display_cols], use_container_width=True)
        with c2:
            if not df.empty:
                s = st.selectbox("選股看圖", df['完整代號'] + " " + df['名稱'])
                row = df[df['完整代號']==s.split()[0]].iloc[0]
                plot_chart(row['完整代號'], row['名稱'])
    else:
        if os.path.exists("welcome.jpg"):
            st.image("welcome.jpg", width=300)

if __name__ == "__main__":
    try:
        main_app()
    except Exception as e:
        st.error("⚠️ 系統發生暫時性錯誤")
        st.warning("👉 建議解決方案：請點擊右下角 'Manage app' -> 選擇 'Reboot app' 即可恢復。")
        with st.expander("查看錯誤詳細資訊 (給工程師看)"):
            st.code(traceback.format_exc())

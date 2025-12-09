import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import os

# --- 1. 網頁設定 ---
VER = "ver5.1_Stable"
st.set_page_config(page_title=f"🍍 旺來-台股生命線({VER})", layout="wide")

# --- 2. 核心功能區 ---

# ★★★ 修正重點：加入 show_spinner=False 避免喚醒時報錯 ★★★
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list():
    try:
        tse = twstock.twse
        otc = twstock.tpex
        stock_dict = {}
        exclude = ['金融保險業', '存託憑證']
        for code, info in tse.items():
            if info.type == '股票' and info.group not in exclude:
                stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code}
        for code, info in otc.items():
            if info.type == '股票' and info.group not in exclude:
                stock_dict[f"{code}.TWO"] = {'name': info.name, 'code': code}
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

# --- 策略擂台 (含動態出場) ---
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

                            # 訊號
                            cond_near = (lp <= m200v * 1.03) and (lp >= m200v * 0.90)
                            cond_up = (cp > m200v)
                            basic = cond_near and cond_up
                            
                            trend_up = (m200v > float(m200.iloc[idx-20]))
                            vol_dbl = (vol > p_vol * 1.5)
                            
                            # 皇冠: 多頭排列
                            crown = (cp > m20v) and (m20v > m60v) and (m60v > m200v) and trend_up

                            # 浴火重生
                            treasure = False
                            if idx >= 7:
                                rc, rm = c.iloc[idx-7:idx+1], m200.iloc[idx-7:idx+1]
                                if rc.iloc[-1] > rm.iloc[-1] and (rc.iloc[:-1] < rm.iloc[:-1]).any():
                                    treasure = True

                            if not basic and not treasure and not crown: continue
                                
                            # 績效
                            if idx + 20 < len(c):
                                # 靜態
                                ret_s = (float(c.iloc[idx+20]) - cp) / cp * 100
                                win_s = ret_s > 0

                                # 動態
                                exit_d = float(c.iloc[idx+20])
                                for fi in range(1, 21):
                                    fidx = idx + fi
                                    if fidx >= len(c): break
                                    if float(h.iloc[fidx]) >= cp * 1.10: # 停利
                                        exit_d = cp * 1.10
                                        break
                                    if float(c.iloc[fidx]) < float(m200.iloc[fidx]) * 0.99: # 停損
                                        exit_d = float(c.iloc[fidx])
                                        break
                                ret_d = (exit_d - cp) / cp * 100
                                win_d = ret_d > 0
                                
                                raw_signals.append({

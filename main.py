#!/usr/bin/env python3
"""
Professional Forex H1 Telegram bot - Render-ready
- Uses TwelveData for H1 data (you can change provider)
- Multiple strategies incl. SKK-style confluence
- Sends recommendations to Telegram group/chat
- Keeps a tiny webserver for Render health checks (so it stays alive)
USAGE:
- Put your secrets in environment variables on Render:
  TG_BOT_TOKEN, TG_CHAT_ID, DATA_API_KEY
- Deploy to Render (or any host) with start command: python main.py
"""
import os
import time
import math
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
from flask import Flask
from threading import Thread

# Telegram bot (using raw requests to avoid version issues)
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or "REPLACE_WITH_YOUR_TOKEN"
TG_CHAT_ID   = os.getenv("TG_CHAT_ID") or "REPLACE_WITH_CHAT_ID"
DATA_API_KEY = os.getenv("DATA_API_KEY") or "REPLACE_WITH_DATA_API_KEY"

if "REPLACE_WITH" in TG_BOT_TOKEN or "REPLACE_WITH" in TG_CHAT_ID or "REPLACE_WITH" in DATA_API_KEY:
    print("Warning: environment variables not set. Edit main.py or set env vars TG_BOT_TOKEN, TG_CHAT_ID, DATA_API_KEY.")
# Settings
PAIRS = ["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD","NZD/USD","XAU/USD"]
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
INTERVAL = "1h"
HIST_OHLCV = 600
REQUEST_SLEEP = 1.2
BACKTEST_LOOKAHEAD = 24
BACKTEST_MAX_SIGNALS = 150

# tiny webserver for healthcheck (Render / UptimeRobot)
app = Flask('health')
@app.route('/')
def home():
    return "Forex bot is alive"

def run_web():
    app.run(host='0.0.0.0', port=8080)

# --- Data fetcher ---
def fetch_ohlcv_twelvedata(symbol, interval=INTERVAL, outputsize=HIST_OHLCV):
    params = {
        "symbol": symbol.replace("/",""),
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": DATA_API_KEY
    }
    try:
        r = requests.get(TWELVEDATA_URL, params=params, timeout=20)
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Network error fetching {symbol}: {e}")
    if "values" not in data:
        raise RuntimeError(f"Failed fetch {symbol}: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0
    return df

# --- Indicators ---
def compute_indicators(df):
    df = df.copy()
    df["sma50"] = ta.sma(df["close"], length=50)
    df["sma200"] = ta.sma(df["close"], length=200)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_middle"] = bb["BBM_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    return df

def fib_levels(df, lookback=200):
    recent = df.tail(lookback)
    hi = recent["high"].max()
    lo = recent["low"].min()
    diff = hi - lo if hi != lo else 1e-9
    levels = {
        "0.0": hi,
        "0.236": hi - 0.236 * diff,
        "0.382": hi - 0.382 * diff,
        "0.5": hi - 0.5 * diff,
        "0.618": hi - 0.618 * diff,
        "1.0": lo
    }
    return levels

# --- strategies ---
def strat_sma(df):
    last = df.iloc[-1]
    if pd.isna(last["sma50"]) or pd.isna(last["sma200"]): return 0
    return 1 if last["sma50"] > last["sma200"] else -1

def strat_ema(df):
    last = df.iloc[-1]
    if pd.isna(last["ema21"]) or pd.isna(last["ema50"]): return 0
    return 1 if last["ema21"] > last["ema50"] else -1

def strat_rsi(df):
    last = df.iloc[-1]
    if pd.isna(last["rsi14"]): return 0
    if last["rsi14"] < 30: return 1
    if last["rsi14"] > 70: return -1
    return 0

def strat_macd(df):
    last = df.iloc[-1]
    if pd.isna(last["macd"]) or pd.isna(last["macd_signal"]): return 0
    return 1 if last["macd"] > last["macd_signal"] else -1

def strat_bollinger(df):
    last = df.iloc[-1]
    if pd.isna(last["bb_upper"]) or pd.isna(last["bb_lower"]): return 0
    price = last["close"]
    if price > last["bb_upper"]: return 1
    if price < last["bb_lower"]: return -1
    return 0

def strat_candles(df):
    if len(df) < 3: return 0
    last = df.iloc[-1]; prev = df.iloc[-2]
    # bullish engulfing
    if prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open'] and last['open'] < prev['close']:
        return 1
    # bearish engulfing
    if prev['close'] > prev['open'] and last['close'] < last['open'] and last['open'] > prev['close'] and last['close'] < prev['open']:
        return -1
    return 0

def strat_skk(df):
    fibs = fib_levels(df, lookback=200)
    price = df["close"].iloc[-1]
    tolerance = 0.002
    for lvl in fibs.values():
        if abs(price - lvl) / price <= tolerance:
            rsi = df["rsi14"].iloc[-1]
            macd = df["macd"].iloc[-1]; macd_signal = df["macd_signal"].iloc[-1]
            if not math.isnan(rsi) and rsi < 50 and macd < macd_signal:
                return -1
            if not math.isnan(rsi) and rsi > 50 and macd > macd_signal:
                return 1
    return 0

WEIGHTS = {"sma":1.0,"ema":1.0,"rsi":0.9,"macd":0.9,"bb":0.8,"candle":0.8,"skk":1.2}

def combined_signal(df):
    fibs = fib_levels(df, lookback=200)
    votes = []
    votes.append(WEIGHTS["sma"] * strat_sma(df))
    votes.append(WEIGHTS["ema"] * strat_ema(df))
    votes.append(WEIGHTS["rsi"] * strat_rsi(df))
    votes.append(WEIGHTS["macd"] * strat_macd(df))
    votes.append(WEIGHTS["bb"] * strat_bollinger(df))
    votes.append(WEIGHTS["candle"] * strat_candles(df))
    votes.append(WEIGHTS["skk"] * strat_skk(df))
    score = sum(votes)
    abs_max = sum(abs(w) for w in WEIGHTS.values())
    confidence = min(100, round((abs(score) / abs_max) * 100, 1))
    if score >= 1.2:
        return "BUY", confidence, fibs
    if score <= -1.2:
        return "SELL", confidence, fibs
    return "HOLD", confidence, fibs

def compute_tp_sl(price, direction, atr, fibs=None):
    if atr is None or math.isnan(atr) or atr == 0:
        atr = max(0.001 * abs(price), 0.0001)
    if direction == "BUY":
        sl = price - 1.0 * atr; tp = price + 2.0 * atr
    elif direction == "SELL":
        sl = price + 1.0 * atr; tp = price - 2.0 * atr
    else:
        return None, None
    if fibs:
        candidates = sorted(fibs.values())
        if direction == "BUY":
            for c in candidates:
                if c > price and (c - price) <= 10 * atr:
                    tp = c; break
        else:
            for c in reversed(candidates):
                if c < price and (price - c) <= 10 * atr:
                    tp = c; break
    return round(float(tp), 6), round(float(sl), 6)

def backtest_pair(df):
    df2 = compute_indicators(df)
    signals_results = []; signals_count = 0
    for i in range(220, len(df2) - BACKTEST_LOOKAHEAD):
        window = df2.iloc[:i+1].copy()
        decision, conf, fibs = combined_signal(window)
        if decision == "HOLD": continue
        entry = float(df2["close"].iloc[i])
        atr = float(df2["atr14"].iloc[i]) if not math.isnan(df2["atr14"].iloc[i]) else 0.001 * entry
        tp, sl = compute_tp_sl(entry, decision, atr, fibs)
        future = df2.iloc[i+1 : i+1+BACKTEST_LOOKAHEAD]
        hit = None
        for _, row in future.iterrows():
            if decision == "BUY":
                if row["high"] >= tp: hit = "TP"; break
                if row["low"] <= sl: hit = "SL"; break
            elif decision == "SELL":
                if row["low"] <= tp: hit = "TP"; break
                if row["high"] >= sl: hit = "SL"; break
        if hit:
            signals_results.append(1 if hit == "TP" else 0); signals_count += 1
        if signals_count >= BACKTEST_MAX_SIGNALS: break
    if len(signals_results) == 0: return None
    success_rate = sum(signals_results) / len(signals_results)
    return round(success_rate * 100, 2)

def telegram_send(text):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def analyze_and_send(pair):
    try:
        df = fetch_ohlcv_twelvedata(pair)
        df = compute_indicators(df)
        decision, confidence, fibs = combined_signal(df)
        price = float(df["close"].iloc[-1])
        atr = float(df["atr14"].iloc[-1]) if not math.isnan(df["atr14"].iloc[-1]) else 0.001 * price
        tp, sl = compute_tp_sl(price, decision, atr, fibs)
        hist_success = backtest_pair(df)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = (f"<b>{pair}</b>\n{now_utc}\nSignal: <b>{decision}</b> (conf: {confidence}%)\nPrice: {price}\n"
               f"TP: {tp if tp else '-'}\nSL: {sl if sl else '-'}\nATR: {round(atr,6)}\n"
               f"Historical succ: {hist_success if hist_success is not None else 'N/A'}%\nFib(0.618): {round(fibs['0.618'],6)}")
        ok, resp = telegram_send(msg)
        return ok, resp
    except Exception as e:
        return False, str(e)

def run_loop():
    # align to top of hour
    now = datetime.now(timezone.utc)
    seconds_to_next_hour = ((now.replace(minute=0, second=10, microsecond=0) + timedelta(hours=1)) - now).total_seconds()
    if seconds_to_next_hour > 0:
        print(f"Sleeping {int(seconds_to_next_hour)}s until top of next hour...")
        time.sleep(seconds_to_next_hour)
    while True:
        start = time.time()
        for pair in PAIRS:
            ok, resp = analyze_and_send(pair)
            if ok:
                print(f"Sent {pair}")
            else:
                print(f"Failed {pair}: {resp}")
            time.sleep(REQUEST_SLEEP)
        elapsed = time.time() - start
        sleep_for = max(0, 3600 - elapsed)
        print(f"Cycle done. Sleeping {int(sleep_for)}s until next hour.")
        time.sleep(sleep_for)

if __name__ == "__main__":
    # start health webserver in background
    t = Thread(target=run_web, daemon=True); t.start()
    run_loop()

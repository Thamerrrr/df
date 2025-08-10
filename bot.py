"""
Forex + Gold H1 analysis Telegram bot (Professional version)
- Sources: TwelveData (time_series, 1h)
- Strategies combined: SMA, EMA, RSI, MACD, Bollinger, Fibonacci proximity, ATR-based TP/SL
- Backtest: simple forward-checking over recent history to estimate hit-rate for signals
- Designed to run continuously (Railway / VPS)
"""

import os
import time
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
import math

# -----------------------------
# CONFIG - امني: يفضّل استخدام ENV VARs
# -----------------------------
# أفضل: ضع هذه المتغيرات في Environment variables في Railway (Project -> Variables)
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TG_CHAT_ID    = os.getenv("TG_CHAT_ID") or "YOUR_CHAT_ID_HERE"
DATA_API_KEY  = os.getenv("DATA_API_KEY") or "YOUR_TWELVEDATA_API_KEY_HERE"

# -----------------------------
# PAIRS & SETTINGS
# -----------------------------
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "XAU/USD"
]

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
INTERVAL = "1h"
HIST_OHLCV = 600     # عدد الشموع المسحوبة (كافٍ لباكتيست قصير ومتوسط)
REQUEST_SLEEP = 1.2  # ثانية بين طلبات API (احترام rate limits)
BACKTEST_LOOKAHEAD = 24  # ساعات للنظر للأمام لكل إشارة أثناء الباكتيست
BACKTEST_MAX_SIGNALS = 200

# -----------------------------
# Utility: fetch OHLCV from TwelveData
# -----------------------------
def fetch_ohlcv_twelvedata(symbol, interval=INTERVAL, outputsize=HIST_OHLCV):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": DATA_API_KEY
    }
    r = requests.get(TWELVEDATA_URL, params=params, timeout=20)
    data = r.json()
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

# -----------------------------
# Indicators computation
# -----------------------------
def compute_indicators(df):
    df = df.copy()
    # SMAs & EMAs
    df["sma50"] = ta.sma(df["close"], length=50)
    df["sma200"] = ta.sma(df["close"], length=200)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)
    # RSI
    df["rsi14"] = ta.rsi(df["close"], length=14)
    # MACD
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    # Bollinger
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_middle"] = bb["BBM_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
    # ATR
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    return df

# -----------------------------
# Fibonacci simple levels from recent swing
# -----------------------------
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

# -----------------------------
# Single-strategy signals (binary / soft)
# -----------------------------
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
    # breakout above upper => bullish, below lower => bearish
    if price > last["bb_upper"]: return 1
    if price < last["bb_lower"]: return -1
    return 0

def strat_fib_proximity(df, fibs, tolerance=0.0015):
    # If price is very close to a fib level, vote for mean-reversion
    price = df["close"].iloc[-1]
    diffs = {k: abs(price - v) for k,v in fibs.items()}
    key = min(diffs, key=diffs.get)
    dist = diffs[key]
    if dist / price <= tolerance:
        lvl = fibs[key]
        return 1 if price < lvl else -1
    return 0

# -----------------------------
# Combine strategies (weighted vote)
# -----------------------------
STRAT_WEIGHTS = {
    "sma": 1.0,
    "ema": 1.0,
    "rsi": 0.9,
    "macd": 0.9,
    "bb": 0.8,
    "fib": 0.7
}

def combined_signal(df):
    fibs = fib_levels(df, lookback=200)
    votes = []
    votes.append(STRAT_WEIGHTS["sma"] * strat_sma(df))
    votes.append(STRAT_WEIGHTS["ema"] * strat_ema(df))
    votes.append(STRAT_WEIGHTS["rsi"] * strat_rsi(df))
    votes.append(STRAT_WEIGHTS["macd"] * strat_macd(df))
    votes.append(STRAT_WEIGHTS["bb"] * strat_bollinger(df))
    votes.append(STRAT_WEIGHTS["fib"] * strat_fib_proximity(df, fibs))
    score = sum(votes)
    abs_max = sum(abs(w) for w in STRAT_WEIGHTS.values())
    confidence = min(100, round((abs(score) / abs_max) * 100, 1))
    if score >= 1.2:
        return "BUY", confidence, fibs
    if score <= -1.2:
        return "SELL", confidence, fibs
    return "HOLD", confidence, fibs

# -----------------------------
# TP / SL computation (ATR + fib)
# -----------------------------
def compute_tp_sl(price, direction, atr, fibs=None):
    if atr is None or math.isnan(atr) or atr == 0:
        atr = max(0.001 * abs(price), 0.0001)
    if direction == "BUY":
        sl = price - 1.0 * atr
        tp = price + 2.0 * atr
    elif direction == "SELL":
        sl = price + 1.0 * atr
        tp = price - 2.0 * atr
    else:
        return None, None
    if fibs:
        candidates = [v for v in fibs.values()]
        if direction == "BUY":
            above = sorted([c for c in candidates if c > price])
            if above:
                tp_candidate = above[0]
                if (tp_candidate - price) <= 10 * atr:
                    tp = tp_candidate
        else:
            below = sorted([c for c in candidates if c < price], reverse=True)
            if below:
                tp_candidate = below[0]
                if (price - tp_candidate) <= 10 * atr:
                    tp = tp_candidate
    return round(float(tp), 6), round(float(sl), 6)

# -----------------------------
# Backtest simple forward-checking to compute success rate
# -----------------------------
def backtest_pair(df):
    df2 = compute_indicators(df)
    signals_results = []
    signals_count = 0
    for i in range(210, len(df2) - BACKTEST_LOOKAHEAD):
        window = df2.iloc[:i+1].copy()
        decision, conf, fibs = combined_signal(window)
        if decision == "HOLD":
            continue
        entry = float(df2["close"].iloc[i])
        atr = float(df2["atr14"].iloc[i]) if not math.isnan(df2["atr14"].iloc[i]) else 0.001 * entry
        tp, sl = compute_tp_sl(entry, decision, atr, fibs)
        future = df2.iloc[i+1 : i+1+BACKTEST_LOOKAHEAD]
        hit = None
        for _, row in future.iterrows():
            if decision == "BUY":
                if row["high"] >= tp:
                    hit = "TP"; break
                if row["low"] <= sl:
                    hit = "SL"; break
            elif decision == "SELL":
                if row["low"] <= tp:
                    hit = "TP"; break
                if row["high"] >= sl:
                    hit = "SL"; break
        if hit:
            signals_results.append(1 if hit == "TP" else 0)
            signals_count += 1
        if signals_count >= BACKTEST_MAX_SIGNALS:
            break
    if len(signals_results) == 0:
        return None
    success_rate = sum(signals_results) / len(signals_results)
    return round(success_rate * 100, 2)

# -----------------------------
# Telegram send
# -----------------------------
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

# -----------------------------
# Analyze & send for one pair
# -----------------------------
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
        msg = f"<b>{pair}</b>\n{now_utc}\nSignal: <b>{decision}</b> (conf: {confidence}%)\nPrice: {price}\nTP: {tp if tp else '-'}\nSL: {sl if sl else '-'}\nATR: {round(atr,6)}\nHistorical success (recent): {hist_success if hist_success is not None else 'N/A'}%\nFib(0.618): {round(fibs['0.618'],6)}"
        ok, resp = send_telegram(msg)
        return ok, resp
    except Exception as e:
        return False, str(e)

# -----------------------------
# Main loop: align to the top of each hour, then run across pairs
# -----------------------------
def sleep_until_next_hour():
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
    wait = (next_hour - now).total_seconds()
    if wait > 0:
        print(f"Sleeping {int(wait)}s until top of next hour ({next_hour.isoformat()})")
        time.sleep(wait)

def run_continuous():
    print("Starting professional Forex bot (H1).")
    sleep_until_next_hour()
    while True:
        cycle_start = datetime.now(timezone.utc)
        print(f"Cycle start: {cycle_start.isoformat()}")
        for pair in PAIRS:
            ok, resp = analyze_and_send(pair)
            if ok:
                print(f"Sent {pair}")
            else:
                print(f"Failed {pair}: {resp}")
            time.sleep(REQUEST_SLEEP)
        sleep_until_next_hour()

if __name__ == "__main__":
    if "YOUR_TELEGRAM_TOKEN" in TG_BOT_TOKEN or TG_BOT_TOKEN.strip()=="":
        raise SystemExit("Set TG_BOT_TOKEN (env var or edit code).")
    run_continuous()

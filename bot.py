"""
Pro Scalper Bot
- Timeframes: 1m (main), uses rolling stats to detect spikes
- Indicators: EMA(9,21), RSI(14), VWAP, ATR(14)
- Extra checks: Volume spike (vs avg), Orderbook imbalance (bid/ask depth)
- Risk mgmt: account balance, risk% per trade, SL from ATR, TP by R:R ratio
- Sends Telegram alerts with signal, TP, SL, confidence score

Save as pro_scalper_bot.py and run: python pro_scalper_bot.py
"""

import requests
import time
import math
import csv
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ================== CONFIG (edit as needed) ==================
import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Enable test mode if Telegram credentials are missing
TEST_MODE = not (TELEGRAM_TOKEN and CHAT_ID)
if TEST_MODE:
    print("⚠️  TEST MODE: Running without Telegram notifications")
    print("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in Replit secrets to enable alerts")
    TELEGRAM_TOKEN = "test_token"
    CHAT_ID = "test_chat"
SYMBOLS = ["DOGEUSDT","XRPUSDT","LTCUSDT","BTCUSDT","ETHUSDT"]  # change list to monitor
MAIN_INTERVAL = "1m"   # scalping timeframe
FETCH_CANDLES = 200    # number of candles to fetch for indicators
CHECK_INTERVAL = 20    # seconds between checks (choose 20-60 for scalping)
ACCOUNT_BALANCE_USDT = 1000.0
RISK_PER_TRADE_PERCENT = 1.0  # percent of account balance risked per trade
RR_RATIO = 1.8  # take-profit risk-reward ratio
VOLUME_SPIKE_MULTIPLIER = 1.6  # last candle volume > multiplier * avg volume -> spike
ORDERBOOK_DEPTH = 10  # levels to sum for imbalance
MIN_CONFIDENCE_TO_ALERT = 75  # percent - increased for stronger signals only
LOG_CSV = "pro_scalper_signals.csv"
# ============================================================

HEADERS = {"User-Agent":"pro-scalper-bot/1.0"}

# -------------------- Utility / Market Data --------------------
def send_telegram(message: str):
    if TEST_MODE:
        print(f"📧 [TEST MODE] Would send: {message}")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=8)
        if r.status_code != 200:
            print("Telegram send error:", r.status_code, r.text)
    except Exception as e:
        print("Telegram exception:", e)

def fetch_klines(symbol: str, interval: str = "1m", limit: int = 200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=8, headers=HEADERS)
    data = r.json()
    # kline format: [ openTime, open, high, low, close, volume, closeTime, ... ]
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float,
        "open_time": int, "close_time": int
    })
    df['datetime'] = pd.to_datetime(df['close_time'], unit='ms')
    return df

def fetch_orderbook(symbol: str, limit: int = 50):
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(url, params=params, timeout=6, headers=HEADERS)
    return r.json()

# -------------------- Indicators --------------------
def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def vwap(df: pd.DataFrame):
    # vwap over provided candles
    pv = (df['close'] * df['volume']).astype(float)
    return pv.cumsum() / df['volume'].cumsum()

def atr(df: pd.DataFrame, length: int = 14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=length, adjust=False).mean()
    return atr

# -------------------- Signal Logic --------------------
def analyze_symbol(symbol):
    try:
        df = fetch_klines(symbol, MAIN_INTERVAL, FETCH_CANDLES)
    except Exception as e:
        print("Klines fetch error", symbol, e)
        return None

    # compute indicators
    df['close_f'] = df['close'].astype(float)
    df['volume_f'] = df['volume'].astype(float)
    df['EMA9'] = ema(df['close_f'], 9)
    df['EMA21'] = ema(df['close_f'], 21)
    df['RSI14'] = rsi(df['close_f'], 14)
    df['VWAP'] = vwap(df)
    df['ATR14'] = atr(df, 14)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest['close_f'])
    ema9 = float(latest['EMA9'])
    ema21 = float(latest['EMA21'])
    rsi14 = float(latest['RSI14'])
    vwap_now = float(latest['VWAP'])
    atr_now = float(latest['ATR14'])
    last_vol = float(latest['volume_f'])
    avg_vol = float(df['volume_f'][-30:].mean())

    # volume spike
    vol_spike = last_vol > (avg_vol * VOLUME_SPIKE_MULTIPLIER)

    # EMA trend and crossover
    ema_trend = "bull" if ema9 > ema21 else "bear"
    crossover = None
    # check if crossover happened in last candle
    prev_ema9 = float(prev['EMA9'])
    prev_ema21 = float(prev['EMA21'])
    if prev_ema9 <= prev_ema21 and ema9 > ema21:
        crossover = "bullish"
    elif prev_ema9 >= prev_ema21 and ema9 < ema21:
        crossover = "bearish"

    # ENHANCED RSI CONDITIONS - More selective
    rsi_oversold = rsi14 < 30  # Strong oversold
    rsi_overbought = rsi14 > 70  # Strong overbought
    rsi_moderate_buy = 30 <= rsi14 < 45  # Moderate buy zone
    rsi_moderate_sell = 55 < rsi14 <= 70  # Moderate sell zone

    # Orderbook imbalance
    ob = fetch_orderbook(symbol, limit=ORDERBOOK_DEPTH)
    bids = ob.get('bids', [])[:ORDERBOOK_DEPTH]
    asks = ob.get('asks', [])[:ORDERBOOK_DEPTH]
    sum_bids = sum([float(b[1]) for b in bids]) if bids else 0.0
    sum_asks = sum([float(a[1]) for a in asks]) if asks else 0.0
    imbalance = (sum_bids - sum_asks) / (sum_bids + sum_asks + 1e-9)  # -1..1
    # positive imbalance -> more bids (bullish), negative -> more asks (bearish)

    # price change percent last candle vs previous close
    prev_close = float(prev['close_f'])
    price_change_pct = ((price - prev_close) / (prev_close + 1e-9)) * 100

    # ADVANCED SIGNAL FILTERING - Only strongest signals pass
    # Multiple confirmation system for higher accuracy
    score = 50  # base
    reasons = []
    confirmation_count = 0  # Track how many confirmations we have

    # ENHANCED EMA ANALYSIS with trend strength
    ema_gap = abs(ema9 - ema21) / price * 100  # EMA separation as %
    strong_trend = ema_gap > 0.5  # Strong trend if EMAs are well separated
    
    if crossover == "bullish":
        score += 20
        confirmation_count += 1
        reasons.append("🚀 EMA bullish crossover")
        if strong_trend:
            score += 10
            reasons.append("💪 Strong bullish trend")
    elif crossover == "bearish":
        score -= 20
        confirmation_count += 1
        reasons.append("📉 EMA bearish crossover")
        if strong_trend:
            score -= 10
            reasons.append("💪 Strong bearish trend")
    else:
        # Current trend strength bonus
        if ema_trend == "bull" and strong_trend:
            score += 8
            confirmation_count += 1
            reasons.append("📈 Strong bull trend")
        elif ema_trend == "bear" and strong_trend:
            score -= 8
            confirmation_count += 1
            reasons.append("📉 Strong bear trend")

    # ENHANCED RSI SCORING
    if rsi_oversold:
        score += 20
        confirmation_count += 1
        reasons.append("🔴 RSI extremely oversold")
    elif rsi_moderate_buy:
        score += 10
        reasons.append("🟡 RSI in buy zone")
    
    if rsi_overbought:
        score -= 20
        confirmation_count += 1
        reasons.append("🔴 RSI extremely overbought")
    elif rsi_moderate_sell:
        score -= 10
        reasons.append("🟡 RSI in sell zone")

    # ENHANCED VOLUME ANALYSIS
    volume_ratio = last_vol / avg_vol
    if volume_ratio >= 2.5:  # Extreme volume spike
        score += 25
        confirmation_count += 1
        reasons.append(f"🚀 Extreme volume spike ({volume_ratio:.1f}x)")
    elif vol_spike:  # Normal volume spike
        score += 15
        reasons.append(f"📈 Volume spike ({volume_ratio:.1f}x)")

    # ENHANCED ORDERBOOK ANALYSIS
    if imbalance > 0.25:  # Strong bid pressure
        score += 18
        confirmation_count += 1
        reasons.append(f"💪 Strong bid pressure ({imbalance:.2f})")
    elif imbalance > 0.12:
        score += 10
        reasons.append(f"📊 Orderbook bid-heavy ({imbalance:.2f})")
    elif imbalance < -0.25:  # Strong ask pressure
        score -= 18
        confirmation_count += 1
        reasons.append(f"💪 Strong ask pressure ({imbalance:.2f})")
    elif imbalance < -0.12:
        score -= 10
        reasons.append(f"📊 Orderbook ask-heavy ({imbalance:.2f})")

    # ENHANCED MOMENTUM ANALYSIS
    if price_change_pct > 1.0:  # Strong positive momentum
        score += 15
        confirmation_count += 1
        reasons.append(f"🚀 Strong momentum (+{price_change_pct:.2f}%)")
    elif price_change_pct > 0.5:
        score += 8
        reasons.append(f"📈 Good momentum (+{price_change_pct:.2f}%)")
    elif price_change_pct < -1.0:  # Strong negative momentum
        score -= 15
        confirmation_count += 1
        reasons.append(f"📉 Strong decline ({price_change_pct:.2f}%)")
    elif price_change_pct < -0.5:
        score -= 8
        reasons.append(f"📉 Weak momentum ({price_change_pct:.2f}%)")

    # ENHANCED VWAP FILTER
    vwap_distance = ((price - vwap_now) / vwap_now) * 100
    if vwap_distance > 0.5:  # Significantly above VWAP
        score += 12
        confirmation_count += 1
        reasons.append(f"📊 Strong above VWAP (+{vwap_distance:.2f}%)")
    elif price > vwap_now:
        score += 6
        reasons.append(f"📊 Above VWAP (+{vwap_distance:.2f}%)")
    elif vwap_distance < -0.5:  # Significantly below VWAP
        score -= 12
        confirmation_count += 1
        reasons.append(f"📊 Strong below VWAP ({vwap_distance:.2f}%)")
    else:
        score -= 6
        reasons.append(f"📊 Below VWAP ({vwap_distance:.2f}%)")

    # Calculate confidence based on signal strength (distance from neutral)
    # Use absolute value so both strong BUY and SELL signals have high confidence
    confidence = max(0, min(100, int(abs(score - 50) * 2)))  # Scale to 0-100

    # ADVANCED SIGNAL DETERMINATION - Only strongest signals
    action = None
    
    # Require multiple confirmations for high-confidence signals
    min_confirmations = 3 if confidence >= 80 else 2
    
    if confidence >= MIN_CONFIDENCE_TO_ALERT and confirmation_count >= min_confirmations:
        if score >= 75:  # Strong bullish signal
            action = "BUY"
        elif score <= 25:  # Strong bearish signal
            action = "SELL"
        else:
            # Even with good confidence, reject if signal not decisive enough
            reasons.append(f"❌ Rejected: Signal not decisive (score: {score})")
    elif confidence >= MIN_CONFIDENCE_TO_ALERT:
        reasons.append(f"❌ Rejected: Only {confirmation_count}/{min_confirmations} confirmations")

    # Risk & SL/TP calc (use ATR)
    sl = None; tp = None; position_size = None
    if action:
        # SL distance = ATR * multiplier (scalpers prefer small ATR multiple)
        atr_ticks = max(atr_now, 1e-6)
        sl_distance = atr_ticks * 1.0  # 1*ATR for tight SL (scalp)
        if action == "BUY":
            sl = price - sl_distance
            tp = price + sl_distance * RR_RATIO
        else:
            sl = price + sl_distance
            tp = price - sl_distance * RR_RATIO

        # risk amount in USDT
        risk_amount = ACCOUNT_BALANCE_USDT * (RISK_PER_TRADE_PERCENT / 100.0)
        # position size = risk_amount / |entry - sl|
        price_diff = abs(price - sl) if abs(price - sl) > 1e-9 else 1e-9
        position_size = risk_amount / price_diff
        # limit position size sanity
        max_pos = (ACCOUNT_BALANCE_USDT * 0.2) / price  # don't use >20% of account in a single pos
        position_size = min(position_size, max_pos)

    # Compose result
    result = {
        "symbol": symbol.replace("USDT",""),
        "price": price,
        "ema9": ema9, "ema21": ema21, "rsi14": rsi14,
        "vwap": vwap_now, "atr": atr_now,
        "volume": last_vol, "avg_volume": avg_vol,
        "vol_spike": vol_spike,
        "orderbook_imbalance": imbalance,
        "price_change_pct": price_change_pct,
        "confidence": confidence,
        "confirmations": confirmation_count,
        "action": action,
        "sl": sl, "tp": tp, "position_size": position_size,
        "reasons": reasons,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    return result

# -------------------- Logging --------------------
def log_signal(res):
    header = ["timestamp","symbol","action","price","sl","tp","position_size","confidence","reasons","volume","avg_volume","orderbook_imbalance"]
    write_header = False
    try:
        with open(LOG_CSV, 'r', newline='') as f:
            pass
    except FileNotFoundError:
        write_header = True

    with open(LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            res["timestamp"], res["symbol"], res["action"], res["price"],
            res["sl"], res["tp"], res["position_size"], res["confidence"],
            ";".join(res["reasons"]), res["volume"], res["avg_volume"], res["orderbook_imbalance"]
        ])

# -------------------- Main Loop --------------------
def main_loop():
    print("🚀 Pro Scalper Bot started at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    scan_count = 0
    while True:
        try:
            scan_count += 1
            print(f"🔍 Scan #{scan_count} - Analyzing {len(SYMBOLS)} symbols...")
            
            for sym in SYMBOLS:
                res = analyze_symbol(sym)
                if not res:
                    print(f"❌ {sym}: Failed to analyze")
                    continue

                # Show analysis results for transparency
                print(f"📈 {res['symbol']}: Price=${res['price']:.4f}, Confidence={res['confidence']}%, Confirmations={res['confirmations']}")
                if res['reasons']:
                    print(f"   Reasons: {', '.join(res['reasons'][:3])}...")  # Show first 3 reasons

                # If there's an actionable signal, send Telegram alert
                if res["action"]:
                    message = (
                        f"🚨 <b>#{res['symbol']} {res['action']} Signal</b>\n"
                        f"🕜 {res['timestamp']}\n"
                        f"💰 Price: {res['price']:.8f} USDT\n"
                        f"📈 Change(last candle): {res['price_change_pct']:.2f}%\n"
                        f"📊 Volume: {res['volume']:.2f} (avg {res['avg_volume']:.2f})\n"
                        f"🔎 Reasons: {', '.join(res['reasons'])}\n\n"
                        f"🎯 TP: {res['tp']:.8f}\n"
                        f"🛑 SL: {res['sl']:.8f}\n"
                        f"⚖️ Position size (est): {res['position_size']:.6f} {res['symbol']}\n"
                        f"✅ Confidence: {res['confidence']}% | Confirmations: {res['confirmations']}\n"
                        f"Risk per trade: {RISK_PER_TRADE_PERCENT}% of ${ACCOUNT_BALANCE_USDT}\n"
                    )
                    send_telegram(message)
                    log_signal(res)
                    print(f"🚨 SIGNAL GENERATED: {res['symbol']} {res['action']} - Confidence: {res['confidence']}%")

                # small delay per symbol to avoid rate limits
                time.sleep(0.8)
                
            print(f"✅ Scan #{scan_count} completed. Waiting {CHECK_INTERVAL}s for next scan...\n")
        except Exception as e:
            print("❌ Main loop exception:", e)
            import traceback
            traceback.print_exc()
        # Wait before next full scan
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()

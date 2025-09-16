"""
Pro Scalper Bot - Render Optimized
- Timeframes: 1m (main), uses rolling stats to detect spikes
- Indicators: EMA(9,21), RSI(14), VWAP, ATR(14)
- Extra checks: Volume spike (vs avg), Orderbook imbalance (bid/ask depth)
- Risk mgmt: account balance, risk% per trade, SL from ATR, TP by R:R ratio
- Sends Telegram alerts with signal, TP, SL, confidence score

Optimized for Render.com deployment
"""

import requests
import time
import math
import csv
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import os

# ================== CONFIGURATION ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Enable test mode if Telegram credentials are missing
TEST_MODE = not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
if TEST_MODE:
    print("⚠  TEST MODE: Running without Telegram notifications")
    print("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID as environment variables")

SYMBOLS = ["DOGEUSDT", "XRPUSDT", "LTCUSDT", "BTCUSDT", "ETHUSDT"]
MAIN_INTERVAL = "1m"
FETCH_CANDLES = 200
CHECK_INTERVAL = 20
ACCOUNT_BALANCE_USDT = 1000.0
RISK_PER_TRADE_PERCENT = 1.0
RR_RATIO = 1.8
VOLUME_SPIKE_MULTIPLIER = 1.6
ORDERBOOK_DEPTH = 10
MIN_CONFIDENCE_TO_ALERT = 75
LOG_CSV = "pro_scalper_signals.csv"
# ===================================================

HEADERS = {"User-Agent": "pro-scalper-bot/1.0 (Render)"}

# -------------------- Utility Functions --------------------
def send_telegram(message: str):
    """Send Telegram notification with proper error handling"""
    if TEST_MODE:
        print(f"📧 [TEST MODE] Would send: {message}")
        return True
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"Telegram API Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Telegram Send Exception: {e}")
        return False

def fetch_klines(symbol: str, interval: str = "1m", limit: int = 200):
    """Fetch OHLCV data from Binance with error handling"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=10, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume", 
            "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
        ])
        
        # Convert to appropriate data types
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error fetching klines for {symbol}: {e}")
        return None

def fetch_orderbook(symbol: str, limit: int = 50):
    """Fetch orderbook data with error handling"""
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=8, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching orderbook for {symbol}: {e}")
        return {"bids": [], "asks": []}

# -------------------- Technical Indicators --------------------
def ema(series: pd.Series, length: int):
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def vwap(df: pd.DataFrame):
    """Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    return pv.cumsum() / df['volume'].cumsum()

def atr(df: pd.DataFrame, length: int = 14):
    """Average True Range - Fixed Version"""
    try:
        # Ensure we're working with pandas Series
        high = df['high']
        low = df['low'] 
        close = df['close']
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        # Get maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using EMA
        atr_value = true_range.ewm(span=length, adjust=False).mean()
        
        return atr_value
        
    except Exception as e:
        print(f"ATR calculation error: {e}")
        # Return a series of zeros with same length as input
        return pd.Series([0.0] * len(df), index=df.index)

# -------------------- Signal Analysis --------------------
def analyze_symbol(symbol):
    """Comprehensive technical analysis for a symbol"""
    try:
        df = fetch_klines(symbol, MAIN_INTERVAL, FETCH_CANDLES)
        if df is None or len(df) < 50:
            print(f"⚠  Insufficient data for {symbol}")
            return None
            
        # Ensure we have enough data for indicators
        if len(df) < 30:
            print(f"⚠  Not enough data for indicators: {symbol}")
            return None
            
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

    # Calculate technical indicators
    try:
        df['EMA9'] = ema(df['close'], 9)
        df['EMA21'] = ema(df['close'], 21)
        df['RSI14'] = rsi(df['close'], 14)
        df['VWAP'] = vwap(df)
        df['ATR14'] = atr(df, 14)  # This will use the fixed function now
    except Exception as e:
        print(f"Indicator calculation error for {symbol}: {e}")
        return None

    # Get latest values
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest['close'])
    ema9 = float(latest['EMA9'])
    ema21 = float(latest['EMA21'])
    rsi14 = float(latest['RSI14'])
    vwap_now = float(latest['VWAP'])
    atr_now = float(latest['ATR14'])
    last_vol = float(latest['volume'])
    avg_vol = float(df['volume'][-30:].mean())

    # Volume analysis
    vol_spike = last_vol > (avg_vol * VOLUME_SPIKE_MULTIPLIER)
    volume_ratio = last_vol / (avg_vol + 1e-9)

    # EMA analysis
    ema_trend = "bull" if ema9 > ema21 else "bear"
    ema_gap = abs(ema9 - ema21) / (price + 1e-9) * 100
    strong_trend = ema_gap > 0.5

    # Check for EMA crossover
    prev_ema9 = float(prev['EMA9'])
    prev_ema21 = float(prev['EMA21'])
    crossover = None
    if prev_ema9 <= prev_ema21 and ema9 > ema21:
        crossover = "bullish"
    elif prev_ema9 >= prev_ema21 and ema9 < ema21:
        crossover = "bearish"

    # RSI analysis
    rsi_oversold = rsi14 < 30
    rsi_overbought = rsi14 > 70
    rsi_moderate_buy = 30 <= rsi14 < 45
    rsi_moderate_sell = 55 < rsi14 <= 70

    # Orderbook analysis
    ob = fetch_orderbook(symbol, ORDERBOOK_DEPTH)
    bids = ob.get('bids', [])[:ORDERBOOK_DEPTH]
    asks = ob.get('asks', [])[:ORDERBOOK_DEPTH]
    
    sum_bids = sum(float(b[1]) for b in bids) if bids else 0.0
    sum_asks = sum(float(a[1]) for a in asks) if asks else 0.0
    imbalance = (sum_bids - sum_asks) / (sum_bids + sum_asks + 1e-9)

    # Price momentum
    prev_close = float(prev['close'])
    price_change_pct = ((price - prev_close) / (prev_close + 1e-9)) * 100

    # VWAP analysis
    vwap_distance = ((price - vwap_now) / (vwap_now + 1e-9)) * 100

    # Signal scoring system
    score = 50  # Neutral starting point
    reasons = []
    confirmation_count = 0

    # EMA Scoring
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
    elif strong_trend:
        if ema_trend == "bull":
            score += 8
            reasons.append("📈 Strong bull trend")
        else:
            score -= 8
            reasons.append("📉 Strong bear trend")

    # RSI Scoring
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

    # Volume Scoring
    if volume_ratio >= 2.5:
        score += 25
        confirmation_count += 1
        reasons.append(f"🚀 Extreme volume spike ({volume_ratio:.1f}x)")
    elif vol_spike:
        score += 15
        reasons.append(f"📈 Volume spike ({volume_ratio:.1f}x)")

    # Orderbook Scoring
    if imbalance > 0.25:
        score += 18
        confirmation_count += 1
        reasons.append(f"💪 Strong bid pressure ({imbalance:.2f})")
    elif imbalance > 0.12:
        score += 10
        reasons.append(f"📊 Orderbook bid-heavy ({imbalance:.2f})")
    elif imbalance < -0.25:
        score -= 18
        confirmation_count += 1
        reasons.append(f"💪 Strong ask pressure ({imbalance:.2f})")
    elif imbalance < -0.12:
        score -= 10
        reasons.append(f"📊 Orderbook ask-heavy ({imbalance:.2f})")

    # Momentum Scoring
    if price_change_pct > 1.0:
        score += 15
        confirmation_count += 1
        reasons.append(f"🚀 Strong momentum (+{price_change_pct:.2f}%)")
    elif price_change_pct > 0.5:
        score += 8
        reasons.append(f"📈 Good momentum (+{price_change_pct:.2f}%)")
    elif price_change_pct < -1.0:
        score -= 15
        confirmation_count += 1
        reasons.append(f"📉 Strong decline ({price_change_pct:.2f}%)")
    elif price_change_pct < -0.5:
        score -= 8
        reasons.append(f"📉 Weak momentum ({price_change_pct:.2f}%)")

    # VWAP Scoring
    if vwap_distance > 0.5:
        score += 12
        confirmation_count += 1
        reasons.append(f"📊 Strong above VWAP (+{vwap_distance:.2f}%)")
    elif price > vwap_now:
        score += 6
        reasons.append(f"📊 Above VWAP (+{vwap_distance:.2f}%)")
    elif vwap_distance < -0.5:
        score -= 12
        confirmation_count += 1
        reasons.append(f"📊 Strong below VWAP ({vwap_distance:.2f}%)")
    else:
        score -= 6
        reasons.append(f"📊 Below VWAP ({vwap_distance:.2f}%)")

    # Calculate final confidence
    confidence = max(0, min(100, int(abs(score - 50) * 2)))

    # Signal determination
    action = None
    min_confirmations = 3 if confidence >= 80 else 2
    
    if confidence >= MIN_CONFIDENCE_TO_ALERT and confirmation_count >= min_confirmations:
        if score >= 75:
            action = "BUY"
        elif score <= 25:
            action = "SELL"

    # Risk management calculations
    sl = tp = position_size = None
    if action:
        atr_ticks = max(atr_now, 1e-6)
        sl_distance = atr_ticks * 1.0
        
        if action == "BUY":
            sl = price - sl_distance
            tp = price + sl_distance * RR_RATIO
        else:
            sl = price + sl_distance
            tp = price - sl_distance * RR_RATIO

        risk_amount = ACCOUNT_BALANCE_USDT * (RISK_PER_TRADE_PERCENT / 100.0)
        price_diff = abs(price - sl)
        position_size = risk_amount / price_diff if price_diff > 1e-9 else 0
        max_pos = (ACCOUNT_BALANCE_USDT * 0.2) / price
        position_size = min(position_size, max_pos)

    # Compose result
    return {
        "symbol": symbol.replace("USDT", ""),
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

# -------------------- Logging --------------------
def log_signal(res):
    """Log signals to CSV file"""
    try:
        file_exists = os.path.exists(LOG_CSV)
        with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "symbol", "action", "price", "sl", "tp",
                    "position_size", "confidence", "reasons", "volume",
                    "avg_volume", "orderbook_imbalance"
                ])
            writer.writerow([
                res["timestamp"], res["symbol"], res["action"], res["price"],
                res["sl"], res["tp"], res["position_size"], res["confidence"],
                ";".join(res["reasons"]), res["volume"], res["avg_volume"],
                res["orderbook_imbalance"]
            ])
    except Exception as e:
        print(f"Error logging signal: {e}")

# -------------------- Main Loop --------------------
def main_loop():
    """Main trading loop"""
    print(f"🚀 Pro Scalper Bot started at {datetime.now(timezone.utc)}")
    print(f"📊 Monitoring symbols: {', '.join(SYMBOLS)}")
    print(f"⏰ Check interval: {CHECK_INTERVAL} seconds")
    
    if TEST_MODE:
        print("🔶 Running in TEST MODE - No Telegram notifications")
    else:
        print("✅ Telegram notifications ENABLED")
    
    scan_count = 0
    
    while True:
        try:
            scan_count += 1
            print(f"\n🔍 Scan #{scan_count} - {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
            
            for symbol in SYMBOLS:
                result = analyze_symbol(symbol)
                if not result:
                    continue

                # Display analysis results
                status_icon = "✅" if result['action'] else "➖"
                print(f"{status_icon} {result['symbol']}: "
                      f"${result['price']:.4f} | "
                      f"Conf: {result['confidence']}% | "
                      f"Confirmations: {result['confirmations']}")

                # Send alert for actionable signals
                if result["action"]:
                    message = (
                        f"🚨 <b>#{result['symbol']} {result['action']} Signal</b>\n"
                        f"🕜 {result['timestamp']}\n"
                        f"💰 Price: {result['price']:.8f}\n"
                        f"📈 Change: {result['price_change_pct']:.2f}%\n"
                        f"📊 Volume: {result['volume']:.2f} (avg {result['avg_volume']:.2f})\n"
                        f"🔎 Reasons: {', '.join(result['reasons'][:3])}\n\n"
                        f"🎯 TP: {result['tp']:.8f}\n"
                        f"🛑 SL: {result['sl']:.8f}\n"
                        f"⚖ Size: {result['position_size']:.6f} {result['symbol']}\n"
                        f"✅ Confidence: {result['confidence']}%\n"
                        f"🔢 Confirmations: {result['confirmations']}"
                    )
                    
                    if send_telegram(message):
                        print(f"📢 Telegram alert sent for {result['symbol']}")
                    log_signal(result)
                    print(f"🚨 SIGNAL: {result['symbol']} {result['action']} "
                          f"(Conf: {result['confidence']}%)")

                time.sleep(0.5)  # Rate limiting between symbols
                
            print(f"⏳ Next scan in {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(60)  # Wait longer on critical errors

if __name__ == "__main__":
    main_loop()

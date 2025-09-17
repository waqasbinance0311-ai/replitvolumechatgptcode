# pro_scalper_bot.py
import os
import requests
import time
import csv
import pandas as pd
from datetime import datetime
import pytz

# ================== CONFIG ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID", "YOUR_CHAT_ID")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT", "BNBUSDT"]
INTERVAL = "1m"
CANDLES = 100
CHECK_INTERVAL = 60           # seconds between scans
CONF_THRESHOLD = 70           # min confidence to alert
ACCOUNT_BALANCE_USDT = 1000
RISK_PER_TRADE_PERCENT = 1.0
RR_RATIO = 1.8
LOG_CSV = "signals_log.csv"
USER_AGENT = "pro-price-action-bot/1.0"

HEADERS = {"User-Agent": USER_AGENT}
last_alerts = {}  # keep track to avoid duplicates (key -> minute string)
karachi = pytz.timezone("Asia/Karachi")

# ================== HELPERS / MARKET DATA ==================
def get_klines(symbol, interval="1m", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ])
        # convert types
        numeric = ["open","high","low","close","volume","open_time","close_time"]
        for c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_klines] {symbol} error:", e)
        return None

def get_orderbook_imbalance(symbol, limit=20):
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=8, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        bids = sum(float(b[1]) for b in data.get("bids", []))
        asks = sum(float(a[1]) for a in data.get("asks", []))
        if bids + asks == 0:
            return 0.0
        return (bids - asks) / (bids + asks)  # -1 .. 1
    except Exception as e:
        print(f"[get_orderbook] {symbol} error:", e)
        return 0.0

# ================== INDICATORS / PRICE ACTION ==================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def detect_structure(df, lookback=5):
    """Simple structure detection using last few highs/lows."""
    if len(df) < lookback + 1:
        return "neutral"
    highs = df['high'].tail(lookback).reset_index(drop=True)
    lows = df['low'].tail(lookback).reset_index(drop=True)
    # Compare last two swings
    if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
        return "bull"
    if highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
        return "bear"
    return "neutral"

# ================== SIGNAL LOGGING ==================
def log_signal(row: dict):
    header = ["timestamp","symbol","action","price","sl","tp","pos_size","confidence","reasons","volume","avg_volume","orderbook"]
    file_exists = os.path.exists(LOG_CSV)
    try:
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([
                row.get("time"), row.get("symbol"), row.get("action"), row.get("price"),
                row.get("sl"), row.get("tp"), row.get("pos_size"), row.get("confidence"),
                ";".join(row.get("reasons", [])), row.get("volume"), row.get("avg_volume"),
                row.get("orderbook")
            ])
    except Exception as e:
        print("[log_signal] error:", e)

# ================== CORE ANALYSIS ==================
def analyze_symbol(symbol):
    df = get_klines(symbol, INTERVAL, CANDLES)
    if df is None or len(df) < 30:
        return None

    price = float(df['close'].iloc[-1])
    volume = float(df['volume'].iloc[-1])
    avg_volume = float(df['volume'].tail(20).mean())

    # indicators
    ema50 = ema(df['close'], 50).iloc[-1]
    rsi14 = rsi(df['close'], 14).iloc[-1]
    structure = detect_structure(df, lookback=5)
    ob = get_orderbook_imbalance(symbol)

    # score system (weights)
    score = 50
    reasons = []

    # Price Action (structure)
    if structure == "bull":
        score += 12
        reasons.append("Bullish structure (HH-HL)")
    elif structure == "bear":
        score -= 12
        reasons.append("Bearish structure (LH-LL)")

    # EMA trend
    if price > ema50:
        score += 8
        reasons.append("Above EMA50")
    else:
        score -= 8
        reasons.append("Below EMA50")

    # RSI
    if rsi14 < 35:
        score += 8
        reasons.append("RSI oversold")
    elif rsi14 > 65:
        score -= 8
        reasons.append("RSI overbought")

    # Volume spike (recent candle vs avg 20)
    if avg_volume > 0 and volume > avg_volume * 1.5:
        score += 12
        reasons.append(f"Volume spike ({volume/ (avg_volume+1e-9):.2f}x)")

    # Orderbook imbalance
    if ob > 0.15:
        score += 10
        reasons.append(f"Orderbook bid-heavy ({ob:.2f})")
    elif ob < -0.15:
        score -= 10
        reasons.append(f"Orderbook ask-heavy ({ob:.2f})")

    # Normalize
    confidence = max(0, min(100, int(score)))

    # Determine action bias: require multiple confirmations
    action = None
    # bullish bias
    bullish_points = 0
    bearish_points = 0
    if structure == "bull":
        bullish_points += 1
    if price > ema50:
        bullish_points += 1
    if rsi14 < 50:
        bullish_points += 1
    if ob > 0.05:
        bullish_points += 1
    if volume > avg_volume * 1.2:
        bullish_points += 1

    if structure == "bear":
        bearish_points += 1
    if price < ema50:
        bearish_points += 1
    if rsi14 > 50:
        bearish_points += 1
    if ob < -0.05:
        bearish_points += 1
    if volume < avg_volume * 0.8:
        bearish_points += 1

    if bullish_points >= 3:
        action = "BUY"
    elif bearish_points >= 3:
        action = "SELL"

    # final confidence check
    if action is None or confidence < CONF_THRESHOLD:
        return None

    # SL/TP via ATR-like simple method (mean HL of last 14)
    atr_est = (df['high'] - df['low']).tail(14).mean()
    if atr_est is None or atr_est <= 0:
        atr_est = max( (df['high'] - df['low']).tail(5).mean(), 0.0001)

    if action == "BUY":
        sl = price - atr_est
        tp = price + atr_est * RR_RATIO
    else:
        sl = price + atr_est
        tp = price - atr_est * RR_RATIO

    # position sizing (risk %)
    risk_amount = ACCOUNT_BALANCE_USDT * (RISK_PER_TRADE_PERCENT / 100.0)
    price_diff = abs(price - sl)
    pos_size = (risk_amount / price_diff) if price_diff > 1e-9 else 0
    max_pos = (ACCOUNT_BALANCE_USDT * 0.2) / price
    pos_size = min(pos_size, max_pos)

    result = {
        "symbol": symbol.replace("USDT", ""),
        "price": price,
        "volume": volume,
        "avg_volume": avg_volume,
        "rsi": round(rsi14,2),
        "ema50": round(ema50,6),
        "structure": structure,
        "orderbook": round(ob,3),
        "confidence": confidence,
        "action": action,
        "sl": round(sl,6),
        "tp": round(tp,6),
        "pos_size": round(pos_size,6),
        "reasons": reasons,
        "time": datetime.now(karachi).strftime("%Y-%m-%d %H:%M:%S PKT")
    }
    return result

# ================== ALERTS ==================
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=8)
    except Exception as e:
        print("[send_telegram] error:", e)

def format_and_send(sig):
    if not sig:
        return
    key = f"{sig['symbol']}_{sig['action']}"
    minute_key = datetime.now(karachi).strftime("%Y-%m-%d %H:%M")  # minute resolution

    # avoid duplicate alert in same minute
    if last_alerts.get(key) == minute_key:
        return
    last_alerts[key] = minute_key

    msg = (
        f"🚨 #{sig['symbol']} {sig['action']} Signal\n"
        f"🕜 {sig['time']}\n"
        f"💰 Price: {sig['price']:.6f} USDT\n"
        f"📈 RSI: {sig['rsi']}\n"
        f"📊 Volume: {sig['volume']:.2f} (avg {sig['avg_volume']:.2f})\n"
        f"🔎 Reasons: {', '.join(sig['reasons'])}\n\n"
        f"🎯 TP: {sig['tp']:.6f}\n"
        f"🛑 SL: {sig['sl']:.6f}\n"
        f"⚖ Position size: {sig['pos_size']:.6f} {sig['symbol']}\n"
        f"✅ Confidence: {sig['confidence']}%\n"
        f"Risk per trade: {RISK_PER_TRADE_PERCENT}% of ${ACCOUNT_BALANCE_USDT}"
    )

    send_telegram_message(msg)
    log_signal({
        "time": sig['time'],
        "symbol": sig['symbol'],
        "action": sig['action'],
        "price": sig['price'],
        "sl": sig['sl'],
        "tp": sig['tp'],
        "pos_size": sig['pos_size'],
        "confidence": sig['confidence'],
        "reasons": sig['reasons'],
        "volume": sig['volume'],
        "avg_volume": sig['avg_volume'],
        "orderbook": sig['orderbook']
    })
    print(f"[ALERT] {sig['symbol']} {sig['action']} | Conf: {sig['confidence']}%")

# ================== MAIN LOOP ==================
def main():
    print("🚀 Pro Price Action Scalper (with logging) started...")
    print("Monitoring:", ", ".join(SYMBOLS))
    while True:
        for sym in SYMBOLS:
            try:
                sig = analyze_symbol(sym)
                if sig and sig.get("action"):
                    format_and_send(sig)
            except Exception as e:
                print(f"[main] {sym} error:", e)
            time.sleep(1)  # small pause between symbols to reduce rate usage
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()


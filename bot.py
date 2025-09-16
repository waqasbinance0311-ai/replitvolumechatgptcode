import os
import requests
import time

# ================== CONFIGURATION ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8023108538:AAE51wAdhjHSv6TQOYBBe7RS0jIrOTRoOcs")
CHAT_ID = os.getenv("CHAT_ID", "5969642968")

SYMBOLS = ["DOGEUSDT", "XRPUSDT", "LTCUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

ACCOUNT_BALANCE_USDT = 10000     # Total Balance
RISK_PER_TRADE_PERCENT = 0.07    # Risk per trade (%)
CHECK_INTERVAL = 60              # Check every X seconds
# ===================================================

# -------------------- Binance API --------------------
def get_binance_data(symbol):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

# -------------------- Signal Analysis --------------------
def analyze_symbol(symbol):
    data = get_binance_data(symbol)
    if not data or "lastPrice" not in data:
        return None

    price = float(data["lastPrice"])
    volume = float(data["volume"])
    price_change = float(data["priceChangePercent"])

    # --- Simple trading logic ---
    action = None
    if price_change > 2:       # Example rule → Buy if pump > 2%
        action = "BUY"
    elif price_change < -2:    # Example rule → Sell if dump < -2%
        action = "SELL"

    # --- Risk management (fixed SL/TP) ---
    sl = tp = position_size = None
    if action:
        if action == "BUY":
            sl = price - 2.5
            tp = price + 8.5
        else:  # SELL
            sl = price + 2.5
            tp = price - 8.5

        risk_amount = ACCOUNT_BALANCE_USDT * (RISK_PER_TRADE_PERCENT / 100.0)
        price_diff = abs(price - sl)
        position_size = risk_amount / price_diff if price_diff > 1e-9 else 0
        max_pos = (ACCOUNT_BALANCE_USDT * 0.2) / price
        position_size = min(position_size, max_pos)

    return {
        "symbol": symbol,
        "price": price,
        "volume": volume,
        "change": price_change,
        "action": action,
        "sl": sl,
        "tp": tp,
        "position_size": position_size
    }

# -------------------- Telegram Alert --------------------
def send_alert(signal):
    if not signal or not signal["action"]:
        return

    msg = (
        f"📊 Symbol: {signal['symbol']}\n"
        f"💵 Price: {signal['price']:.4f}\n"
        f"📈 24h Change: {signal['change']:.2f}%\n"
        f"📦 Volume: {signal['volume']:.2f}\n\n"
        f"🚨 Action: {signal['action']}\n"
        f"🎯 Entry: {signal['price']:.2f}\n"
        f"🛑 SL: {signal['sl']:.2f}\n"
        f"✅ TP: {signal['tp']:.2f}\n"
        f"📏 Position Size: {signal['position_size']:.2f} units"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        print(f"✅ Alert sent for {signal['symbol']}")
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# -------------------- Main Loop --------------------
def main():
    print("🚀 Bot started. Checking signals...")
    while True:
        for symbol in SYMBOLS:
            signal = analyze_symbol(symbol)
            if signal and signal["action"]:
                send_alert(signal)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

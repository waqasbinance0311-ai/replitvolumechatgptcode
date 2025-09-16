
Crypto Trading Bot
Overview
A Python-based cryptocurrency scalping bot that analyzes crypto markets and sends trading signals via Telegram. The bot monitors multiple crypto symbols and uses technical indicators to generate buy/sell signals with confidence scores.

Current State
Status: ✅ Successfully imported and running
Environment: Python 3.11 with all dependencies installed
Workflow: Configured to run as a console application
Main File: bot.py
Recent Changes
2025-09-16: Initial project import and major enhancements completed
Installed Python 3.11 and all required dependencies
Fixed Procfile reference from main.py to bot.py
SECURITY FIX: Moved hardcoded Telegram credentials to environment variables
ENHANCED SIGNAL FILTERING: Major upgrade to trading signal accuracy
Increased minimum confidence threshold from 55% to 75%
Added multi-confirmation system requiring 2-3 confirmations for signals
Enhanced EMA analysis with trend strength detection
Improved RSI conditions with extreme oversold/overbought zones
Advanced volume analysis with extreme spike detection (2.5x+ volume)
Sophisticated orderbook imbalance analysis with strong pressure detection
Enhanced momentum analysis with strong positive/negative movement detection
Advanced VWAP filtering with distance-based scoring
Fixed confidence calculation to allow both strong BUY and SELL signals
Added test mode for development without Telegram credentials
Project Architecture
Files Structure
bot.py - Main application file containing the trading bot logic
requirements.txt - Python dependencies (telegram-bot, requests, pandas, ta)
Procfile - Deployment configuration for running the bot
pro_scalper_signals.csv - Generated log file for trading signals (gitignored)
Key Features
Timeframes: 1-minute scalping with rolling statistics
Indicators: EMA(9,21), RSI(14), VWAP, ATR(14)
Analysis: Volume spike detection, orderbook imbalance analysis
Risk Management: Account balance tracking, stop-loss/take-profit calculations
Notifications: Telegram alerts with detailed signal information
Enhanced Configuration
The bot is configured in bot.py with the following enhanced settings:

Monitors: DOGE, XRP, LTC, BTC, ETH (USDT pairs)
Check interval: 20 seconds
Risk per trade: 1% of account balance
Risk-reward ratio: 1.8
Minimum confidence for alerts: 75% (increased for stronger signals)
Multi-confirmation requirements: 2-3 confirmations needed
Advanced filtering thresholds:
Strong trend: EMA separation > 0.5%
Extreme RSI: < 30 (oversold) or > 70 (overbought)
Extreme volume spike: > 2.5x average volume
Strong orderbook imbalance: > 25% bid/ask pressure
Strong momentum: > 1% price change per candle
Significant VWAP distance: > 0.5% from VWAP
Technical Setup
Language: Python 3.11
Dependencies: Installed via pip from requirements.txt
Workflow: Console application running continuously
Data Sources: Binance API for market data and orderbook
Notifications: Telegram Bot API
User Preferences
None specified yet
Required Secrets (Optional)
The bot can run in test mode without Telegram credentials for development. To enable live alerts, set these environment variables in Replit Secrets:

TELEGRAM_TOKEN - Your Telegram bot token from @BotFather
TELEGRAM_CHAT_ID - The chat ID where alerts will be sent
Test Mode: If credentials are not set, the bot runs in test mode and prints alerts to console instead.

Setup Instructions
Create a Telegram bot via @BotFather and get your bot token
Get your chat ID by messaging your bot and checking the Telegram API
Add both values to your Replit Secrets
The bot will start automatically once the secrets are configured
Notes
✅ Security: Telegram credentials are now properly secured via environment variables
It's designed for educational/demo purposes and monitors real market data
LSP shows false positive import errors, but all dependencies work correctly at runtime

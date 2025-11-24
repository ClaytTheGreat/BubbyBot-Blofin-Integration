# BubbyBot-Blofin User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Setting Up Blofin Account](#setting-up-blofin-account)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Bot](#running-the-bot)
6. [Understanding Signals](#understanding-signals)
7. [Risk Management](#risk-management)
8. [Monitoring Trades](#monitoring-trades)
9. [Troubleshooting](#troubleshooting)

## Getting Started

BubbyBot-Blofin is an automated trading bot that executes trades on the Blofin exchange based on technical analysis from Market Cipher, Lux Algo, and Frankie Candles indicators.

### Prerequisites
- Python 3.8 or higher
- Blofin account (demo or live)
- Basic understanding of cryptocurrency trading
- Understanding of leverage and margin trading

## Setting Up Blofin Account

### 1. Create Blofin Account

1. Visit [Blofin.com](https://www.blofin.com/)
2. Click "Sign Up"
3. Complete registration with email verification
4. Complete KYC verification (for live trading)

### 2. Enable Demo Trading (Recommended)

1. Log into your Blofin account
2. Navigate to Demo Trading section
3. Activate demo account (usually starts with 10,000 USDT)

### 3. Create API Keys

**For Demo Trading:**
1. Go to Account → API Management
2. Switch to Demo Trading environment
3. Click "Create API Key"
4. Set API Key name (e.g., "BubbyBot Demo")
5. Set permissions: **READ** and **TRADE** only
6. Set IP whitelist (optional but recommended)
7. Create passphrase (save this securely!)
8. Save your API Key, Secret Key, and Passphrase

**For Live Trading:**
- Same process but in Live Trading environment
- **IMPORTANT**: Start with demo trading first!

### 4. Security Recommendations

- ✅ Enable IP whitelist
- ✅ Use strong passphrase
- ✅ Only enable READ and TRADE permissions
- ✅ Never share API credentials
- ✅ Store credentials securely
- ❌ Do NOT enable TRANSFER permission

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/BubbyBot-Blofin-Integration.git
cd BubbyBot-Blofin-Integration
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Logs Directory

```bash
mkdir logs
```

## Configuration

### Step 1: Create Environment File

```bash
cp .env.example .env
```

### Step 2: Edit Environment File

Open `.env` in a text editor and add your Blofin API credentials:

```bash
# Blofin API Credentials
BLOFIN_API_KEY=your_api_key_here
BLOFIN_SECRET_KEY=your_secret_key_here
BLOFIN_PASSPHRASE=your_passphrase_here

# Trading Configuration
DEMO_MODE=true
MAX_LEVERAGE=50
DEFAULT_MARGIN_MODE=isolated
```

### Step 3: Review Configuration

Edit `config/blofin_config.py` to customize:

**Trading Settings:**
```python
TRADING_CONFIG = {
    "default_leverage": 50,      # Adjust based on risk tolerance
    "max_leverage": 50,
    "default_margin_mode": "isolated",
    "position_mode": "net"
}
```

**Risk Management:**
```python
RISK_MANAGEMENT_CONFIG = {
    "max_position_size_pct": 0.05,  # 5% max per trade
    "max_daily_loss_pct": 0.10,     # 10% daily loss limit
    "default_stop_loss_pct": 0.02,  # 2% stop loss
    "default_take_profit_pct": 0.06 # 6% take profit
}
```

## Running the Bot

### Demo Mode (Recommended for First Time)

```bash
python src/main_blofin.py --demo --mode interactive
```

This will:
- Connect to Blofin demo environment
- Use demo funds (no real money)
- Run in interactive mode for manual control

### Interactive Mode Commands

Once running, you'll see a menu:

```
Commands:
  1 - Show account info
  2 - Execute test signal (BTC-USDT)
  3 - Execute test signal (ETH-USDT)
  4 - Monitor positions
  5 - Check risk limits
  6 - Close all positions
  7 - Show trading stats
  q - Quit
```

### Example Session

```bash
# Start the bot
python src/main_blofin.py --demo --mode interactive

# Check account balance
Enter command: 1

# Execute a test trade on BTC
Enter command: 2

# Monitor the position
Enter command: 4

# Check if within risk limits
Enter command: 5

# Close all positions when done
Enter command: 6

# Quit
Enter command: q
```

## Understanding Signals

### Signal Components

A trading signal includes:

```python
TradingSignal(
    instrument="BTC-USDT",      # Trading pair
    side="buy",                 # buy or sell
    confidence=0.85,            # 0-1 (85% confidence)
    entry_price=50000.0,        # Entry price
    stop_loss=49000.0,          # Stop loss (2% below)
    take_profit=53000.0,        # Take profit (6% above)
    timeframe="15m",            # Timeframe analyzed
    pattern_type="divergence"   # Pattern detected
)
```

### Confidence Levels

- **0.9-1.0 (High)**: 50x leverage, high conviction
- **0.8-0.9 (Medium)**: 35x leverage, good setup
- **0.7-0.8 (Low)**: 20x leverage, acceptable setup
- **Below 0.7**: Signal rejected, not traded

### Signal Sources

1. **Market Cipher**
   - Momentum waves
   - Divergences (bullish/bearish)
   - Trend strength

2. **Lux Algo**
   - Order blocks (support/resistance)
   - Market structure breaks
   - Premium/discount zones

3. **Frankie Candles**
   - Volume profile
   - Price action patterns
   - Volume divergences

## Risk Management

### Mandatory Stop Loss

**Every trade MUST have a stop loss** - this is non-negotiable for capital preservation.

Default stop loss: 2% from entry
- Buy at $50,000 → Stop loss at $49,000
- Sell at $50,000 → Stop loss at $51,000

### Take Profit Targets

Default take profit: 6% from entry (3:1 risk/reward ratio)
- Buy at $50,000 → Take profit at $53,000
- Sell at $50,000 → Take profit at $47,000

### Position Sizing

Maximum position size: 5% of account balance

Example with $10,000 account:
- Maximum position value: $500
- With 50x leverage: $25,000 position size
- Contracts: $25,000 / $50,000 = 0.5 BTC

### Daily Loss Limit

Trading automatically halts if daily loss exceeds 10% of account balance.

Example with $10,000 account:
- Daily loss limit: $1,000
- If losses reach $1,000, bot stops trading
- Resets next day

### Leverage Management

Leverage is dynamically adjusted based on signal confidence:

| Confidence | Leverage | Risk Level |
|------------|----------|------------|
| 0.9-1.0    | 50x      | High       |
| 0.8-0.9    | 35x      | Medium     |
| 0.7-0.8    | 20x      | Low        |

## Monitoring Trades

### Real-Time Monitoring

Use command `4` to monitor positions:

```
Position: BTC-USDT
  Side: long
  Size: 0.5 contracts
  Entry: 50000.00
  Current: 50500.00
  PnL: 25.00 USDT (+1.00%)
  Leverage: 50x
  Liquidation: 49500.00
```

### Key Metrics to Watch

1. **Unrealized PnL**: Current profit/loss
2. **PnL %**: Percentage gain/loss
3. **Liquidation Price**: Price at which position is liquidated
4. **Margin Usage**: How much margin is being used

### Account Status

Use command `1` to view account status:

```
Total Equity: 10250.00 USDT
Available: 9500.00 USDT
Margin Used: 750.00 USDT
Unrealized PnL: 25.00 USDT
Open Positions: 1
Margin Usage: 7.32%
PnL %: 0.24%
```

### Risk Checks

Use command `5` to check risk limits:

```
✅ All risk limits are within acceptable range
```

Or if violated:

```
⚠️ RISK LIMIT VIOLATION ⚠️
  • Daily loss limit exceeded: -12.50% < -10%
🛑 Closing all positions due to daily loss limit
```

## Troubleshooting

### Common Issues

#### 1. "Signature verification failed"

**Cause**: Incorrect API credentials or signature generation

**Solution**:
- Verify API key, secret, and passphrase in `.env`
- Check for extra spaces or newlines
- Ensure system time is synchronized
- Regenerate API keys if needed

#### 2. "Insufficient balance"

**Cause**: Not enough available balance for trade

**Solution**:
- Check available balance (command `1`)
- Reduce position size
- Reduce leverage
- Close existing positions to free up margin

#### 3. "Order rejected: minimum size"

**Cause**: Order size below minimum (usually 0.1 contracts)

**Solution**:
- Increase position size
- Increase account balance
- Adjust position sizing in config

#### 4. "Rate limit exceeded"

**Cause**: Too many API requests in short time

**Solution**:
- Reduce trading frequency
- Wait a few minutes before retrying
- Implement request throttling

#### 5. "Connection timeout"

**Cause**: Network issues or API downtime

**Solution**:
- Check internet connection
- Verify Blofin API status
- Retry after a few minutes
- Check firewall settings

### Getting Help

If you encounter issues:

1. Check the logs: `logs/bubbybot_blofin.log`
2. Review error messages carefully
3. Consult Blofin API documentation
4. Create a GitHub issue with:
   - Error message
   - Log excerpt
   - Steps to reproduce

## Best Practices

### 1. Start with Demo Trading

Always test in demo mode first:
```bash
python src/main_blofin.py --demo --mode interactive
```

### 2. Start Small

When moving to live trading:
- Start with minimum position sizes
- Use lower leverage (20-30x)
- Trade only 1-2 instruments initially

### 3. Monitor Regularly

- Check positions at least every few hours
- Review daily performance
- Analyze winning and losing trades

### 4. Respect Risk Limits

- Never override stop losses
- Never exceed daily loss limits
- Never risk more than 5% per trade

### 5. Keep Learning

- Review trade history
- Analyze what worked and what didn't
- Adjust strategy based on results
- Stay updated on market conditions

### 6. Security

- Never share API credentials
- Use IP whitelisting
- Rotate API keys regularly
- Keep software updated

## Advanced Usage

### Live Trading

**WARNING**: Live trading uses real money!

```bash
python src/main_blofin.py --live --mode interactive
```

You'll be prompted to confirm:
```
⚠️ WARNING: LIVE TRADING MODE ENABLED ⚠️
This will use REAL MONEY on Blofin exchange!
Type 'YES' to confirm live trading:
```

### Automated Mode (Future)

Fully automated trading (coming soon):
```bash
python src/main_blofin.py --demo --mode automated
```

This will:
- Continuously analyze markets
- Generate signals automatically
- Execute trades without manual intervention
- Monitor and manage positions

## Performance Tracking

### View Trading Statistics

Use command `7`:

```
Trading Statistics
Total Trades: 25
Wins: 18
Losses: 7
Win Rate: 72.00%
```

### Log Analysis

Review detailed logs:
```bash
tail -f logs/bubbybot_blofin.log
```

## Conclusion

BubbyBot-Blofin is a powerful trading tool, but remember:

- **Trading involves risk** - never trade with money you can't afford to lose
- **Start with demo** - test thoroughly before going live
- **Use stop losses** - always, without exception
- **Monitor regularly** - stay aware of your positions
- **Keep learning** - improve your strategy over time

Happy trading! 🚀

---

**Need Help?**
- GitHub Issues: [Report a problem](https://github.com/yourusername/BubbyBot-Blofin-Integration/issues)
- Documentation: [Read the docs](../README.md)

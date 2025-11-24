# BubbyBot-Blofin Quick Start Guide 🚀

Get up and running with BubbyBot-Blofin in 5 minutes!

## Prerequisites

- Python 3.8+
- Blofin account (create at [blofin.com](https://www.blofin.com/))
- 5 minutes of your time ⏱️

## Step 1: Get Blofin API Credentials (2 minutes)

### Demo Trading (Recommended)
1. Log into [Blofin](https://www.blofin.com/)
2. Go to **Account → API Management**
3. Switch to **Demo Trading** environment
4. Click **Create API Key**
5. Set permissions: **READ** and **TRADE** only
6. Create a passphrase (save it!)
7. Copy your:
   - API Key
   - Secret Key
   - Passphrase

## Step 2: Install BubbyBot (1 minute)

```bash
# Clone repository
git clone https://github.com/ClaytTheGreat/BubbyBot-Blofin-Integration.git
cd BubbyBot-Blofin-Integration

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create logs directory
mkdir logs
```

## Step 3: Configure API Keys (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or use your favorite editor
```

Add your credentials:
```
BLOFIN_API_KEY=your_api_key_here
BLOFIN_SECRET_KEY=your_secret_key_here
BLOFIN_PASSPHRASE=your_passphrase_here
DEMO_MODE=true
```

Save and exit (Ctrl+X, then Y, then Enter in nano)

## Step 4: Run BubbyBot! (1 minute)

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run in demo mode
python src/main_blofin.py --demo --mode interactive
```

## Step 5: Try Your First Trade!

You'll see a menu like this:

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

### Try These Commands:

1. **Check your balance:**
   ```
   Enter command: 1
   ```

2. **Execute a test trade on BTC:**
   ```
   Enter command: 2
   ```

3. **Monitor your position:**
   ```
   Enter command: 4
   ```

4. **Close the position:**
   ```
   Enter command: 6
   ```

5. **Quit:**
   ```
   Enter command: q
   ```

## What Just Happened?

When you executed command `2`, BubbyBot:
1. ✅ Got current BTC price
2. ✅ Calculated position size (5% of account)
3. ✅ Set stop loss (2% below entry)
4. ✅ Set take profit (6% above entry)
5. ✅ Executed market order with TP/SL
6. ✅ Applied 50x leverage (based on confidence)

All with **mandatory stop loss** for capital protection! 🛡️

## Understanding the Output

### Account Info (Command 1)
```
Total Equity: 10000.00 USDT      ← Your total balance
Available: 9500.00 USDT          ← Available for trading
Margin Used: 500.00 USDT         ← Locked in positions
Unrealized PnL: 25.00 USDT       ← Current profit/loss
Open Positions: 1                 ← Number of open trades
```

### Position Info (Command 4)
```
BTC-USDT:
  Side: long                      ← Buy position
  Size: 0.5 contracts             ← Position size
  Entry: 50000.00                 ← Entry price
  Current: 50500.00               ← Current price
  PnL: 25.00 USDT (+0.50%)       ← Profit/loss
  Leverage: 50x                   ← Leverage used
  Liquidation: 49500.00           ← Liquidation price
```

## Safety Features 🛡️

BubbyBot automatically:
- ✅ Sets stop loss on EVERY trade (mandatory)
- ✅ Limits position size to 5% of account
- ✅ Stops trading if daily loss exceeds 10%
- ✅ Uses isolated margin to limit risk
- ✅ Maintains minimum 2:1 risk/reward ratio

## Next Steps

### 1. Understand the Configuration

Edit `config/blofin_config.py` to customize:
- Leverage (default: 50x)
- Position size (default: 5%)
- Stop loss % (default: 2%)
- Take profit % (default: 6%)

### 2. Test Different Scenarios

Try trading different instruments:
```bash
# In interactive mode
Enter command: 3  # ETH-USDT
```

### 3. Monitor Your Performance

```bash
Enter command: 7  # Show trading stats
```

### 4. Read the Full Documentation

- [User Guide](docs/USER_GUIDE.md) - Detailed usage instructions
- [README](README.md) - Complete documentation
- [Architecture](docs/blofin_integration_architecture.md) - Technical details

## Common Issues

### "Signature verification failed"
- Check API credentials in `.env`
- Ensure no extra spaces
- Regenerate API keys if needed

### "Insufficient balance"
- Check available balance (command 1)
- Reduce position size in config
- Close existing positions

### "Connection timeout"
- Check internet connection
- Verify Blofin API status
- Try again in a few minutes

## Going Live ⚠️

**WARNING**: Live trading uses real money!

When you're ready (after extensive demo testing):

```bash
python src/main_blofin.py --live --mode interactive
```

You'll be asked to confirm:
```
⚠️ WARNING: LIVE TRADING MODE ENABLED ⚠️
Type 'YES' to confirm live trading:
```

### Before Going Live:
1. ✅ Test thoroughly in demo mode
2. ✅ Understand all features
3. ✅ Start with small positions
4. ✅ Use lower leverage (20-30x)
5. ✅ Monitor positions regularly
6. ✅ Never risk more than you can afford to lose

## Need Help?

- 📖 [Full Documentation](README.md)
- 📚 [User Guide](docs/USER_GUIDE.md)
- 🐛 [Report Issues](https://github.com/ClaytTheGreat/BubbyBot-Blofin-Integration/issues)

## Tips for Success 💡

1. **Start Small**: Use minimum position sizes initially
2. **Demo First**: Test for at least a week in demo mode
3. **Monitor Regularly**: Check positions every few hours
4. **Respect Stop Losses**: Never override them
5. **Learn Continuously**: Analyze your trades
6. **Stay Disciplined**: Follow your trading plan

## Congratulations! 🎉

You've successfully set up and run your first trade with BubbyBot-Blofin!

Remember:
- 🛡️ Every trade has a stop loss
- 📊 Position sizing is automatic
- ⚡ Leverage is managed intelligently
- 🎯 Risk/reward ratio is maintained

Happy trading! 🚀

---

**Disclaimer**: Trading involves risk. Never trade with money you can't afford to lose. This software is provided "as is" without warranty. Use at your own risk.

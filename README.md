# BubbyBot-Blofin Integration 🤖

**Advanced AI Trading Bot with Blofin Exchange Integration**

BubbyBot is an intelligent cryptocurrency trading bot that combines Market Cipher, Lux Algo, and Frankie Candles analysis with automated execution on the Blofin exchange platform.

## 🌟 Features

### Trading Intelligence
- **Market Cipher Analysis**: Advanced momentum and trend indicators
- **Lux Algo Integration**: Order blocks, market structure, premium/discount zones
- **Frankie Candles**: Volume profile and divergence analysis
- **Confluence Engine**: Multi-indicator signal validation
- **AI Learning System**: Continuously learns from trading patterns

### Blofin Exchange Integration
- **Full API Support**: Complete REST API integration
- **Automatic TP/SL**: Built-in take profit and stop loss on every trade
- **Position Management**: Real-time position monitoring and management
- **Multiple Instruments**: Support for 400+ USDT-M trading pairs
- **Leverage Trading**: Up to 150x leverage with intelligent management
- **Demo Trading**: Full demo environment support for testing

### Risk Management
- **Mandatory Stop Loss**: Every trade includes a stop loss (capital preservation)
- **Position Sizing**: Intelligent position sizing based on account balance
- **Risk/Reward Ratio**: Minimum 2:1 reward to risk ratio
- **Daily Loss Limits**: Automatic trading halt on excessive losses
- **Leverage Management**: Dynamic leverage based on signal confidence

## 📋 Requirements

- Python 3.8+
- Blofin account (demo or live)
- API credentials from Blofin

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BubbyBot-Blofin-Integration.git
cd BubbyBot-Blofin-Integration

# Install dependencies
pip install -r requirements.txt

# Create logs directory
mkdir logs
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Blofin API credentials
nano .env
```

Add your Blofin API credentials:
```
BLOFIN_API_KEY=your_api_key_here
BLOFIN_SECRET_KEY=your_secret_key_here
BLOFIN_PASSPHRASE=your_passphrase_here
```

### 3. Get Blofin API Credentials

1. Go to [Blofin API Management](https://www.blofin.com/account/api-management)
2. Create a new API key
3. Set permissions: **READ** and **TRADE** (do NOT enable TRANSFER for security)
4. Set IP whitelist (recommended)
5. Save your API Key, Secret Key, and Passphrase

### 4. Run BubbyBot

**Interactive Mode (Recommended for testing):**
```bash
python src/main_blofin.py --demo --mode interactive
```

**Live Trading (WARNING: Real money!):**
```bash
python src/main_blofin.py --live --mode interactive
```

## 🎮 Interactive Mode Commands

Once running in interactive mode, you can use these commands:

- `1` - Show account information
- `2` - Execute test signal (BTC-USDT)
- `3` - Execute test signal (ETH-USDT)
- `4` - Monitor open positions
- `5` - Check risk limits
- `6` - Close all positions
- `7` - Show trading statistics
- `q` - Quit

## 📊 Configuration

### Trading Configuration (`config/blofin_config.py`)

```python
TRADING_CONFIG = {
    "default_leverage": 50,
    "max_leverage": 50,
    "default_margin_mode": "isolated",
    "position_mode": "net"
}
```

### Risk Management Configuration

```python
RISK_MANAGEMENT_CONFIG = {
    "max_position_size_pct": 0.05,  # 5% of account per trade
    "max_daily_loss_pct": 0.10,     # 10% max daily loss
    "default_stop_loss_pct": 0.02,  # 2% stop loss
    "default_take_profit_pct": 0.06, # 6% take profit (3:1 R:R)
    "min_risk_reward_ratio": 2.0,
    "mandatory_stop_loss": True
}
```

### Supported Instruments

Primary pairs:
- BTC-USDT
- ETH-USDT
- SOL-USDT

Secondary pairs:
- AVAX-USDT
- DOGE-USDT
- XRP-USDT
- ADA-USDT
- MATIC-USDT

## 🏗️ Architecture

```
BubbyBot-Blofin-Integration/
├── src/
│   ├── blofin/
│   │   ├── api_client.py           # Core Blofin API client
│   │   ├── exchange_adapter.py     # BubbyBot-Blofin adapter
│   │   └── websocket_client.py     # WebSocket (future)
│   ├── analyzers/
│   │   ├── confluence_engine.py    # Signal confluence
│   │   ├── risk_management_system.py
│   │   └── ai_learning.py
│   └── main_blofin.py              # Main entry point
├── config/
│   └── blofin_config.py            # Configuration
├── tests/
│   └── test_blofin_api.py          # Unit tests
├── docs/
│   └── blofin_integration.md       # Integration docs
├── requirements.txt
├── .env.example
└── README.md
```

## 🔐 Security Best Practices

1. **Never commit API keys** to version control
2. **Use IP whitelisting** on Blofin API keys
3. **Start with demo trading** before going live
4. **Use isolated margin mode** to limit risk
5. **Set appropriate leverage limits** (recommended: 20-50x max)
6. **Enable only necessary permissions** (READ + TRADE, not TRANSFER)
7. **Regularly rotate API keys**

## 📈 Trading Strategy

### Signal Generation

BubbyBot generates trading signals based on:

1. **Market Cipher Indicators**
   - Momentum waves
   - Divergences
   - Trend strength

2. **Lux Algo Analysis**
   - Order blocks
   - Market structure breaks
   - Premium/discount zones

3. **Frankie Candles**
   - Volume profile
   - Price action patterns

### Confluence Scoring

Signals require a minimum confluence score of 0.7 (70%) to execute:
- High confidence (0.9+): 50x leverage
- Medium confidence (0.8-0.9): 35x leverage
- Low confidence (0.7-0.8): 20x leverage

### Risk Management

Every trade includes:
- **Stop Loss**: Mandatory, typically 2% from entry
- **Take Profit**: Typically 6% from entry (3:1 R:R)
- **Position Size**: Maximum 5% of account
- **Daily Loss Limit**: Trading halts at 10% daily loss

## 🧪 Testing

### Demo Trading

Always test with demo trading first:

```bash
python src/main_blofin.py --demo --mode interactive
```

### Unit Tests

```bash
pytest tests/
```

## 📝 Logging

All trading activity is logged to:
- Console output (real-time)
- `logs/bubbybot_blofin.log` (persistent)

Log levels:
- INFO: Normal operations
- WARNING: Risk warnings
- ERROR: Execution errors

## 🚨 Risk Warnings

**IMPORTANT**: Cryptocurrency trading involves substantial risk of loss.

- ⚠️ **Never trade with money you cannot afford to lose**
- ⚠️ **High leverage amplifies both gains and losses**
- ⚠️ **Past performance does not guarantee future results**
- ⚠️ **Always use stop losses to protect capital**
- ⚠️ **Start with demo trading to understand the system**
- ⚠️ **Monitor positions regularly**

## 🔧 Troubleshooting

### Common Issues

**1. Signature verification failed**
- Check that API credentials are correct
- Ensure no extra spaces in .env file
- Verify system time is synchronized

**2. Insufficient balance**
- Check available balance in account
- Reduce position size or leverage
- Ensure margin mode is set correctly

**3. Order rejected**
- Check minimum order size (usually 0.1 contracts)
- Verify instrument ID format (e.g., BTC-USDT)
- Ensure leverage is set before placing order

**4. Rate limit exceeded**
- Reduce API call frequency
- Implement request throttling
- Use WebSocket for real-time data

## 📚 Documentation

- [Blofin API Documentation](https://docs.blofin.com/)
- [Integration Architecture](docs/blofin_integration.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Market Cipher for indicator methodology
- Lux Algo for market structure analysis
- Frankie Candles for volume analysis
- Blofin for exchange API

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/BubbyBot-Blofin-Integration/issues)
- Documentation: [Read the docs](docs/)

## ⚡ Roadmap

### Current Version (v1.0)
- ✅ Blofin API integration
- ✅ Order placement with TP/SL
- ✅ Position monitoring
- ✅ Risk management
- ✅ Demo trading support

### Future Enhancements
- [ ] WebSocket real-time data
- [ ] Full analyzer integration
- [ ] Automated signal generation
- [ ] Multi-timeframe analysis
- [ ] Advanced AI learning
- [ ] Performance analytics dashboard
- [ ] Telegram notifications
- [ ] Multiple exchange support

## 📊 Performance Tracking

Track your performance:
```bash
python src/main_blofin.py --demo --mode interactive
# Command 7: Show trading statistics
```

## 🎯 Best Practices

1. **Start Small**: Begin with minimum position sizes
2. **Use Demo First**: Test thoroughly in demo mode
3. **Monitor Regularly**: Check positions frequently
4. **Respect Risk Limits**: Never override risk management rules
5. **Keep Learning**: Analyze winning and losing trades
6. **Stay Disciplined**: Follow your trading plan
7. **Use Stop Losses**: Always, without exception

---

**Disclaimer**: This software is provided "as is" without warranty of any kind. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

**Remember**: The best trade is often the one you don't take. Trade responsibly! 🚀

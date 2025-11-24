"""
Blofin Configuration
All configuration parameters for Blofin integration
"""

# API Configuration
BLOFIN_API_CONFIG = {
    "base_url": "https://openapi.blofin.com",
    "demo_url": "https://demo-trading-openapi.blofin.com",
    "websocket_public": "wss://openapi.blofin.com/ws/public",
    "websocket_private": "wss://openapi.blofin.com/ws/private",
    "timeout": 30,
    "max_retries": 3
}

# Trading Configuration
TRADING_CONFIG = {
    "default_leverage": 50,
    "max_leverage": 50,
    "default_margin_mode": "isolated",  # or "cross"
    "position_mode": "net",  # or "long_short_mode" for hedge mode
    "default_order_type": "market",
    "use_reduce_only": False
}

# Risk Management Configuration
RISK_MANAGEMENT_CONFIG = {
    # Position sizing
    "max_position_size_pct": 0.05,  # 5% of account per trade
    "min_position_size": 0.1,       # Minimum contracts
    
    # Loss limits
    "max_daily_loss_pct": 0.10,     # 10% max daily loss
    "max_weekly_loss_pct": 0.20,    # 20% max weekly loss
    
    # Stop loss and take profit
    "default_stop_loss_pct": 0.02,  # 2% stop loss
    "default_take_profit_pct": 0.06, # 6% take profit (3:1 R:R)
    "min_risk_reward_ratio": 2.0,   # Minimum 2:1 reward to risk
    "use_trailing_stop": False,
    
    # Leverage management
    "leverage_by_confidence": {
        0.9: 50,  # High confidence: 50x
        0.8: 35,  # Medium confidence: 35x
        0.7: 20   # Low confidence: 20x
    },
    
    # Mandatory stop loss (per user requirements)
    "mandatory_stop_loss": True,
    "allow_market_sl": True  # Allow -1 for market price SL
}

# Instrument Configuration
INSTRUMENTS_CONFIG = {
    "primary": [
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT"
    ],
    "secondary": [
        "AVAX-USDT",
        "DOGE-USDT",
        "XRP-USDT",
        "ADA-USDT",
        "MATIC-USDT"
    ],
    "watchlist": [
        "BNB-USDT",
        "DOT-USDT",
        "LINK-USDT",
        "UNI-USDT"
    ]
}

# BubbyBot Signal Configuration
SIGNAL_CONFIG = {
    # Confluence requirements
    "min_confluence_score": 0.7,    # Minimum 0.7 to trade
    "high_confluence_score": 0.85,  # High confidence threshold
    
    # Analyzer weights
    "analyzer_weights": {
        "market_cipher": 0.35,
        "lux_algo": 0.35,
        "frankie_candles": 0.30
    },
    
    # Signal validation
    "require_all_analyzers": False,  # Don't require all analyzers to agree
    "min_analyzers_agree": 2,        # At least 2 analyzers must agree
    
    # Timeframe preferences
    "primary_timeframe": "15m",
    "confirmation_timeframes": ["5m", "1h"],
    
    # Pattern types
    "preferred_patterns": [
        "divergence",
        "order_block",
        "market_structure_break",
        "premium_discount"
    ]
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "check_positions_interval": 60,  # Check positions every 60 seconds
    "update_account_interval": 300,  # Update account status every 5 minutes
    "log_trades": True,
    "log_file": "logs/blofin_trades.log",
    "enable_alerts": True,
    "alert_on_loss_pct": 0.05  # Alert if position loses 5%
}

# WebSocket Configuration
WEBSOCKET_CONFIG = {
    "enabled": False,  # Enable WebSocket for real-time data
    "reconnect_attempts": 5,
    "reconnect_delay": 5,
    "ping_interval": 30,
    "channels": {
        "public": [
            "tickers",
            "books"
        ],
        "private": [
            "positions",
            "orders",
            "account"
        ]
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/bubbybot_blofin.log",
    "console": True,
    "file_logging": True,
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# Demo Trading Configuration
DEMO_CONFIG = {
    "enabled": True,  # Start with demo trading
    "initial_balance": 10000,  # Demo balance
    "paper_trading": False  # Use Blofin's demo environment
}

# Complete configuration dictionary
BLOFIN_CONFIG = {
    "api": BLOFIN_API_CONFIG,
    "trading": TRADING_CONFIG,
    "risk_management": RISK_MANAGEMENT_CONFIG,
    "instruments": INSTRUMENTS_CONFIG,
    "signals": SIGNAL_CONFIG,
    "monitoring": MONITORING_CONFIG,
    "websocket": WEBSOCKET_CONFIG,
    "logging": LOGGING_CONFIG,
    "demo": DEMO_CONFIG
}


def get_config(demo: bool = False) -> dict:
    """
    Get configuration with demo mode option
    
    Args:
        demo: If True, use demo trading configuration
        
    Returns:
        dict: Configuration dictionary
    """
    config = BLOFIN_CONFIG.copy()
    
    if demo:
        config['demo']['enabled'] = True
        config['api']['base_url'] = config['api']['demo_url']
    
    return config


def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid
    """
    # Check mandatory stop loss
    if not config['risk_management']['mandatory_stop_loss']:
        raise ValueError("Mandatory stop loss must be enabled")
    
    # Check risk/reward ratio
    if config['risk_management']['min_risk_reward_ratio'] < 1.0:
        raise ValueError("Risk/reward ratio must be at least 1:1")
    
    # Check position size
    if config['risk_management']['max_position_size_pct'] > 0.10:
        raise ValueError("Max position size should not exceed 10% of account")
    
    # Check leverage
    if config['trading']['max_leverage'] > 150:
        raise ValueError("Max leverage exceeds Blofin limit of 150x")
    
    return True


if __name__ == "__main__":
    # Test configuration
    config = get_config(demo=True)
    print("Configuration loaded successfully")
    print(f"Demo mode: {config['demo']['enabled']}")
    print(f"Mandatory stop loss: {config['risk_management']['mandatory_stop_loss']}")
    print(f"Max leverage: {config['trading']['max_leverage']}")
    
    # Validate
    if validate_config(config):
        print("✅ Configuration is valid")

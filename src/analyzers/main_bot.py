#!/usr/bin/env python3
"""
BubbyBot Enhanced V2 - Main Bot Runner
Complete integration of Market Cipher, Lux Algo, and Frankie Candles analysis
with self-learning AI and comprehensive risk management
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from bubbybot_enhanced_v2 import BubbyBotEnhancedV2, PatternType, PatternResult
from ai_learning import AILearningSystem
from risk_management_system import RiskManagementSystem
from multi_timeframe_engine import MultiTimeframeEngine
from confluence_engine import ConfluenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bubbybot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BubbyBotMainSystem:
    """Main BubbyBot system that orchestrates all components"""
    
    def __init__(self, config_path: str = "config/trading_config.json"):
        self.config = self.load_config(config_path)
        self.running = False
        self.paper_trading = self.config.get('paper_trading', True)
        
        # Initialize components
        self.bubbybot = BubbyBotEnhancedV2()
        self.ai_learning = AILearningSystem()
        self.risk_manager = RiskManagementSystem()
        self.timeframe_engine = MultiTimeframeEngine()
        self.confluence_engine = ConfluenceEngine()
        
        # Trading state
        self.current_positions = {}
        self.trading_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Initialize database
        self.init_database()
        
        logger.info("BubbyBot Enhanced V2 Main System initialized")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "paper_trading": True,
            "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "risk_management": {
                "max_position_size": 0.02,  # 2% of account
                "max_daily_loss": 0.05,     # 5% daily loss limit
                "stop_loss_pct": 0.02,      # 2% stop loss
                "take_profit_pct": 0.06     # 6% take profit (3:1 RR)
            },
            "market_cipher": {
                "enable_green_dots": True,
                "enable_blood_diamonds": True,
                "enable_yellow_diamonds": True,
                "momentum_threshold": 0.7
            },
            "lux_algo": {
                "enable_order_blocks": True,
                "enable_market_structure": True,
                "premium_discount_zones": True
            },
            "frankie_candles": {
                "enable_volume_profile": True,
                "enable_divergences": True,
                "volume_threshold": 1.5
            }
        }
    
    def init_database(self):
        """Initialize SQLite database for storing trading data"""
        conn = sqlite3.connect('bubbybot_main.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL DEFAULT 0,
                pattern_type TEXT,
                confidence REAL,
                timeframe TEXT,
                status TEXT DEFAULT 'open'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                pattern_type TEXT,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                timeframe TEXT,
                confluence_score REAL,
                executed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                max_drawdown REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    async def analyze_market(self, symbol: str) -> Optional[Dict]:
        """Comprehensive market analysis for a symbol"""
        try:
            # Get market data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="30d", interval="1h")
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Multi-timeframe analysis
            timeframe_signals = {}
            for tf in self.config['timeframes']:
                tf_data = self.get_timeframe_data(symbol, tf)
                if tf_data is not None:
                    # Market Cipher analysis
                    mc_signal = await self.bubbybot.analyze_market_cipher(tf_data)
                    
                    # Lux Algo analysis
                    lux_signal = await self.bubbybot.analyze_lux_algo(tf_data)
                    
                    # Frankie Candles analysis
                    fc_signal = await self.bubbybot.analyze_frankie_candles(tf_data)
                    
                    timeframe_signals[tf] = {
                        'market_cipher': mc_signal,
                        'lux_algo': lux_signal,
                        'frankie_candles': fc_signal
                    }
            
            # Calculate confluence score
            confluence_score = self.confluence_engine.calculate_confluence(timeframe_signals)
            
            # Generate trading signal if confluence is strong
            if confluence_score >= 0.7:  # 70% confluence threshold
                signal = await self.generate_trading_signal(symbol, timeframe_signals, confluence_score)
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return None
    
    def get_timeframe_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for specific timeframe"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Map timeframes to yfinance intervals
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1d'
            }
            
            if timeframe not in interval_map:
                return None
            
            # Determine period based on timeframe
            if timeframe in ['1m', '5m']:
                period = '7d'
            elif timeframe in ['15m', '30m', '1h']:
                period = '30d'
            else:
                period = '1y'
            
            data = ticker.history(period=period, interval=interval_map[timeframe])
            return data if not data.empty else None
            
        except Exception as e:
            logger.error(f"Error getting {timeframe} data for {symbol}: {e}")
            return None
    
    async def generate_trading_signal(self, symbol: str, timeframe_signals: Dict, confluence_score: float) -> Dict:
        """Generate a comprehensive trading signal"""
        try:
            # Determine dominant signal direction
            bullish_signals = 0
            bearish_signals = 0
            
            for tf, signals in timeframe_signals.items():
                for indicator, signal in signals.items():
                    if signal and signal.get('direction') == 'bullish':
                        bullish_signals += 1
                    elif signal and signal.get('direction') == 'bearish':
                        bearish_signals += 1
            
            # Determine signal direction
            if bullish_signals > bearish_signals:
                direction = 'bullish'
            elif bearish_signals > bullish_signals:
                direction = 'bearish'
            else:
                return None  # No clear direction
            
            # Get current price
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d", interval="1m")
            current_price = current_data['Close'].iloc[-1]
            
            # Calculate entry, stop loss, and take profit
            if direction == 'bullish':
                entry_price = current_price
                stop_loss = entry_price * (1 - self.config['risk_management']['stop_loss_pct'])
                take_profit = entry_price * (1 + self.config['risk_management']['take_profit_pct'])
            else:
                entry_price = current_price
                stop_loss = entry_price * (1 + self.config['risk_management']['stop_loss_pct'])
                take_profit = entry_price * (1 - self.config['risk_management']['take_profit_pct'])
            
            # Create signal
            signal = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confluence_score': confluence_score,
                'confidence': min(confluence_score * 100, 95),  # Cap at 95%
                'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss),
                'timeframe_signals': timeframe_signals
            }
            
            # Store signal in database
            self.store_signal(signal)
            
            logger.info(f"Generated {direction} signal for {symbol} with {confluence_score:.2f} confluence")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None
    
    def store_signal(self, signal: Dict):
        """Store trading signal in database"""
        try:
            conn = sqlite3.connect('bubbybot_main.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, signal_type, confidence, 
                                   entry_price, stop_loss, take_profit, confluence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'],
                signal['symbol'],
                signal['direction'],
                signal['confidence'],
                signal['entry_price'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['confluence_score']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    async def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on signal (paper trading for now)"""
        try:
            if not self.paper_trading:
                logger.info("Live trading not implemented yet - use paper trading")
                return False
            
            # Paper trading execution
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price']
            
            # Calculate position size based on risk management
            account_balance = 10000.0  # Paper trading balance
            risk_amount = account_balance * self.config['risk_management']['max_position_size']
            stop_distance = abs(entry_price - signal['stop_loss'])
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            if position_size <= 0:
                logger.warning(f"Invalid position size calculated for {symbol}")
                return False
            
            # Create trade record
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': 'buy' if direction == 'bullish' else 'sell',
                'entry_price': entry_price,
                'quantity': position_size,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'confidence': signal['confidence'],
                'status': 'open'
            }
            
            # Store trade
            self.store_trade(trade)
            self.current_positions[symbol] = trade
            
            logger.info(f"Executed paper trade: {direction} {symbol} at {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def store_trade(self, trade: Dict):
        """Store trade in database"""
        try:
            conn = sqlite3.connect('bubbybot_main.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, side, entry_price, quantity, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'],
                trade['symbol'],
                trade['side'],
                trade['entry_price'],
                trade['quantity'],
                trade['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
    
    async def monitor_positions(self):
        """Monitor open positions for exit conditions"""
        try:
            for symbol, position in list(self.current_positions.items()):
                # Get current price
                ticker = yf.Ticker(symbol)
                current_data = ticker.history(period="1d", interval="1m")
                if current_data.empty:
                    continue
                
                current_price = current_data['Close'].iloc[-1]
                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if position['side'] == 'buy':
                    if current_price <= stop_loss:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price >= take_profit:
                        should_exit = True
                        exit_reason = "take_profit"
                else:  # sell
                    if current_price >= stop_loss:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price <= take_profit:
                        should_exit = True
                        exit_reason = "take_profit"
                
                if should_exit:
                    await self.close_position(symbol, current_price, exit_reason)
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    async def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position"""
        try:
            if symbol not in self.current_positions:
                return
            
            position = self.current_positions[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            # Calculate P&L
            if position['side'] == 'buy':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            # Update trade in database
            conn = sqlite3.connect('bubbybot_main.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trades SET exit_price = ?, pnl = ?, status = ?
                WHERE symbol = ? AND status = 'open'
            ''', (exit_price, pnl, 'closed', symbol))
            
            conn.commit()
            conn.close()
            
            # Remove from current positions
            del self.current_positions[symbol]
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
            
            self.performance_metrics['total_pnl'] += pnl
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades'] * 100
            )
            
            logger.info(f"Closed position {symbol} at {exit_price} ({reason}) - P&L: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting BubbyBot Enhanced V2 trading loop")
        
        while self.running:
            try:
                # Analyze each symbol
                for symbol in self.config['symbols']:
                    signal = await self.analyze_market(symbol)
                    
                    if signal and symbol not in self.current_positions:
                        # Execute trade if we have a strong signal and no existing position
                        if signal['confluence_score'] >= 0.75:  # High confidence threshold
                            await self.execute_trade(signal)
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # AI learning update
                if hasattr(self.ai_learning, 'update_learning'):
                    await self.ai_learning.update_learning()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error
    
    def start(self):
        """Start the bot"""
        self.running = True
        logger.info("BubbyBot Enhanced V2 started")
        
        # Start trading loop in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.run_trading_loop())
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
            loop.close()
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("BubbyBot Enhanced V2 stopped")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'running': self.running,
            'paper_trading': self.paper_trading,
            'current_positions': len(self.current_positions),
            'performance': self.performance_metrics,
            'last_update': datetime.now().isoformat()
        }

def main():
    """Main entry point"""
    logger.info("Initializing BubbyBot Enhanced V2")
    
    # Create bot instance
    bot = BubbyBotMainSystem()
    
    try:
        # Start the bot
        bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down BubbyBot Enhanced V2")
        bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        bot.stop()

if __name__ == "__main__":
    main()

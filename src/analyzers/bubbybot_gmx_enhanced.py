#!/usr/bin/env python3
"""
BubbyBot Enhanced V2 with GMX Integration
Complete AI trading system with browser automation for GMX perpetual trading
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
from gmx_automation import GMXIntegrationManager, GMXTradeOrder, TradeDirection, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bubbybot_gmx.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BubbyBotGMXSystem:
    """Enhanced BubbyBot system with GMX integration"""
    
    def __init__(self, config_path: str = "config/trading_config.json", browser_tools=None):
        self.config = self.load_config(config_path)
        self.running = False
        self.browser_tools = browser_tools
        
        # Initialize core components
        self.bubbybot = BubbyBotEnhancedV2()
        self.ai_learning = AILearningSystem()
        self.risk_manager = RiskManagementSystem()
        self.timeframe_engine = MultiTimeframeEngine()
        self.confluence_engine = ConfluenceEngine()
        
        # Initialize GMX integration if browser tools available
        self.gmx_manager = None
        if browser_tools:
            self.gmx_manager = GMXIntegrationManager(browser_tools)
            logger.info("GMX integration enabled")
        else:
            logger.warning("GMX integration disabled - no browser tools provided")
        
        # Trading state
        self.current_positions = {}
        self.gmx_positions = {}
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
            'sharpe_ratio': 0.0,
            'gmx_trades': 0,
            'gmx_pnl': 0.0
        }
        
        # Signal processing
        self.signal_queue = asyncio.Queue()
        self.processing_signals = False
        
        # Initialize database
        self.init_database()
        
        logger.info("BubbyBot GMX Enhanced System initialized")
    
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
        """Get default configuration with GMX settings"""
        return {
            "paper_trading": False,  # GMX integration for live trading
            "gmx_enabled": True,
            "symbols": ["AVAX-USD", "BTC-USD", "ETH-USD", "SOL-USD"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "risk_management": {
                "max_position_size": 0.02,  # 2% of account per trade
                "max_daily_loss": 0.05,     # 5% daily loss limit
                "stop_loss_pct": 0.02,      # 2% stop loss
                "take_profit_pct": 0.06,    # 6% take profit (3:1 RR)
                "max_leverage": 10.0        # Maximum leverage for GMX
            },
            "gmx_settings": {
                "default_leverage": 5.0,
                "collateral_token": "USDC",
                "slippage_tolerance": 0.005,  # 0.5%
                "auto_close_profit": 5.0,     # Auto close at 5% profit
                "auto_close_loss": 2.0        # Auto close at 2% loss
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
        conn = sqlite3.connect('bubbybot_gmx.db')
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
                platform TEXT DEFAULT 'paper',
                leverage REAL DEFAULT 1.0,
                status TEXT DEFAULT 'open'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gmx_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                size REAL NOT NULL,
                leverage REAL NOT NULL,
                entry_price REAL NOT NULL,
                mark_price REAL,
                liquidation_price REAL,
                pnl REAL DEFAULT 0,
                pnl_percentage REAL DEFAULT 0,
                collateral REAL NOT NULL,
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
                executed BOOLEAN DEFAULT FALSE,
                platform TEXT DEFAULT 'paper'
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    async def initialize_gmx(self) -> bool:
        """Initialize GMX integration"""
        if not self.gmx_manager:
            logger.warning("GMX manager not available")
            return False
        
        try:
            success = await self.gmx_manager.initialize()
            if success:
                logger.info("GMX integration initialized successfully")
                
                # Start position monitoring
                asyncio.create_task(self.monitor_gmx_positions())
                
            return success
            
        except Exception as e:
            logger.error(f"Error initializing GMX: {e}")
            return False
    
    async def analyze_market_comprehensive(self, symbol: str) -> Optional[Dict]:
        """Comprehensive market analysis with enhanced confluence scoring"""
        try:
            # Get market data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="30d", interval="1h")
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Multi-timeframe analysis with weighted scoring
            timeframe_signals = {}
            timeframe_weights = {
                '1m': 0.1, '5m': 0.15, '15m': 0.2, '1h': 0.25, '4h': 0.2, '1d': 0.1
            }
            
            total_confluence = 0.0
            total_weight = 0.0
            
            for tf in self.config['timeframes']:
                tf_data = self.get_timeframe_data(symbol, tf)
                if tf_data is not None:
                    # Market Cipher analysis
                    mc_signal = await self.bubbybot.analyze_market_cipher(tf_data)
                    
                    # Lux Algo analysis
                    lux_signal = await self.bubbybot.analyze_lux_algo(tf_data)
                    
                    # Frankie Candles analysis
                    fc_signal = await self.bubbybot.analyze_frankie_candles(tf_data)
                    
                    # Calculate timeframe confluence
                    tf_confluence = self.calculate_timeframe_confluence(mc_signal, lux_signal, fc_signal)
                    
                    timeframe_signals[tf] = {
                        'market_cipher': mc_signal,
                        'lux_algo': lux_signal,
                        'frankie_candles': fc_signal,
                        'confluence': tf_confluence
                    }
                    
                    # Weight the confluence by timeframe importance
                    weight = timeframe_weights.get(tf, 0.1)
                    total_confluence += tf_confluence * weight
                    total_weight += weight
            
            # Calculate overall confluence score
            overall_confluence = total_confluence / total_weight if total_weight > 0 else 0
            
            # Generate trading signal if confluence is strong enough
            if overall_confluence >= 0.75:  # High confidence threshold for GMX
                signal = await self.generate_enhanced_signal(symbol, timeframe_signals, overall_confluence)
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return None
    
    def calculate_timeframe_confluence(self, mc_signal, lux_signal, fc_signal) -> float:
        """Calculate confluence score for a specific timeframe"""
        signals = [mc_signal, lux_signal, fc_signal]
        valid_signals = [s for s in signals if s is not None]
        
        if not valid_signals:
            return 0.0
        
        # Count bullish vs bearish signals
        bullish_count = sum(1 for s in valid_signals if s.get('direction') == 'bullish')
        bearish_count = sum(1 for s in valid_signals if s.get('direction') == 'bearish')
        
        # Calculate confluence based on agreement
        total_signals = len(valid_signals)
        max_agreement = max(bullish_count, bearish_count)
        
        confluence = max_agreement / total_signals if total_signals > 0 else 0
        
        # Boost confluence if all three indicators agree
        if total_signals == 3 and max_agreement == 3:
            confluence = min(confluence * 1.2, 1.0)  # 20% boost, capped at 1.0
        
        return confluence
    
    async def generate_enhanced_signal(self, symbol: str, timeframe_signals: Dict, confluence_score: float) -> Dict:
        """Generate enhanced trading signal with GMX-specific parameters"""
        try:
            # Determine dominant signal direction
            bullish_weight = 0.0
            bearish_weight = 0.0
            
            timeframe_weights = {
                '1m': 0.1, '5m': 0.15, '15m': 0.2, '1h': 0.25, '4h': 0.2, '1d': 0.1
            }
            
            for tf, signals in timeframe_signals.items():
                weight = timeframe_weights.get(tf, 0.1)
                tf_confluence = signals.get('confluence', 0)
                
                for indicator, signal in signals.items():
                    if signal and isinstance(signal, dict):
                        if signal.get('direction') == 'bullish':
                            bullish_weight += weight * tf_confluence
                        elif signal.get('direction') == 'bearish':
                            bearish_weight += weight * tf_confluence
            
            # Determine signal direction
            if bullish_weight > bearish_weight:
                direction = 'bullish'
                signal_strength = bullish_weight
            elif bearish_weight > bullish_weight:
                direction = 'bearish'
                signal_strength = bearish_weight
            else:
                return None  # No clear direction
            
            # Get current price
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d", interval="1m")
            current_price = current_data['Close'].iloc[-1]
            
            # Calculate enhanced entry, stop loss, and take profit
            volatility = current_data['Close'].pct_change().std() * np.sqrt(1440)  # Daily volatility
            
            if direction == 'bullish':
                entry_price = current_price
                # Dynamic stop loss based on volatility and confluence
                stop_loss_pct = max(0.015, min(0.03, volatility * 2))  # 1.5% to 3%
                stop_loss = entry_price * (1 - stop_loss_pct)
                
                # Dynamic take profit based on risk-reward and confluence
                risk_reward_ratio = 2.5 + (confluence_score * 1.5)  # 2.5:1 to 4:1
                take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio
            else:
                entry_price = current_price
                stop_loss_pct = max(0.015, min(0.03, volatility * 2))
                stop_loss = entry_price * (1 + stop_loss_pct)
                
                risk_reward_ratio = 2.5 + (confluence_score * 1.5)
                take_profit = entry_price - (stop_loss - entry_price) * risk_reward_ratio
            
            # Calculate optimal leverage based on confluence and volatility
            base_leverage = self.config['gmx_settings']['default_leverage']
            max_leverage = self.config['risk_management']['max_leverage']
            
            # Higher confluence = higher leverage (within limits)
            leverage_multiplier = 0.5 + (confluence_score * 1.5)  # 0.5x to 2x
            optimal_leverage = min(base_leverage * leverage_multiplier, max_leverage)
            
            # Create enhanced signal
            signal = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confluence_score': confluence_score,
                'signal_strength': signal_strength,
                'confidence': min(confluence_score * 100, 95),  # Cap at 95%
                'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss),
                'optimal_leverage': optimal_leverage,
                'volatility': volatility,
                'timeframe_signals': timeframe_signals,
                'gmx_ready': True
            }
            
            # Store signal in database
            self.store_enhanced_signal(signal)
            
            logger.info(f"Enhanced signal generated: {direction} {symbol} - Confluence: {confluence_score:.3f}, Leverage: {optimal_leverage:.1f}x")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating enhanced signal: {e}")
            return None
    
    def store_enhanced_signal(self, signal: Dict):
        """Store enhanced trading signal in database"""
        try:
            conn = sqlite3.connect('bubbybot_gmx.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, signal_type, confidence, 
                                   entry_price, stop_loss, take_profit, confluence_score, platform)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'],
                signal['symbol'],
                signal['direction'],
                signal['confidence'],
                signal['entry_price'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['confluence_score'],
                'gmx' if signal.get('gmx_ready') else 'paper'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing enhanced signal: {e}")
    
    async def execute_signal_on_gmx(self, signal: Dict) -> bool:
        """Execute a trading signal on GMX"""
        try:
            if not self.gmx_manager:
                logger.warning("GMX manager not available")
                return False
            
            # Check if GMX is ready
            if not signal.get('gmx_ready', False):
                logger.warning("Signal not ready for GMX execution")
                return False
            
            # Process signal through GMX manager
            success = await self.gmx_manager.process_bubbybot_signal(signal)
            
            if success:
                # Update signal as executed
                self.update_signal_execution(signal['symbol'], signal['timestamp'])
                
                # Update performance metrics
                self.performance_metrics['gmx_trades'] += 1
                
                logger.info(f"Signal executed on GMX: {signal['symbol']} {signal['direction']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing signal on GMX: {e}")
            return False
    
    def update_signal_execution(self, symbol: str, timestamp: str):
        """Update signal execution status in database"""
        try:
            conn = sqlite3.connect('bubbybot_gmx.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE signals SET executed = TRUE 
                WHERE symbol = ? AND timestamp = ?
            ''', (symbol, timestamp))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating signal execution: {e}")
    
    async def monitor_gmx_positions(self):
        """Monitor GMX positions and update database"""
        while self.running:
            try:
                if self.gmx_manager:
                    positions = await self.gmx_manager.gmx_automation.get_current_positions()
                    
                    for position in positions:
                        # Update position in database
                        self.update_gmx_position(position)
                        
                        # Check for auto-close conditions
                        await self.check_auto_close_conditions(position)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring GMX positions: {e}")
                await asyncio.sleep(60)
    
    def update_gmx_position(self, position):
        """Update GMX position in database"""
        try:
            conn = sqlite3.connect('bubbybot_gmx.db')
            cursor = conn.cursor()
            
            # Check if position exists
            cursor.execute('SELECT id FROM gmx_positions WHERE symbol = ? AND status = "open"', (position.symbol,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing position
                cursor.execute('''
                    UPDATE gmx_positions SET 
                    mark_price = ?, pnl = ?, pnl_percentage = ?, 
                    liquidation_price = ?, timestamp = ?
                    WHERE id = ?
                ''', (
                    position.mark_price, position.pnl, position.pnl_percentage,
                    position.liquidation_price, datetime.now().isoformat(), existing[0]
                ))
            else:
                # Insert new position
                cursor.execute('''
                    INSERT INTO gmx_positions 
                    (timestamp, symbol, direction, size, leverage, entry_price, 
                     mark_price, liquidation_price, pnl, pnl_percentage, collateral)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(), position.symbol, position.direction.value,
                    position.size, position.leverage, position.entry_price,
                    position.mark_price, position.liquidation_price,
                    position.pnl, position.pnl_percentage, position.collateral
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating GMX position: {e}")
    
    async def check_auto_close_conditions(self, position):
        """Check if position should be auto-closed"""
        try:
            auto_close_profit = self.config['gmx_settings']['auto_close_profit']
            auto_close_loss = self.config['gmx_settings']['auto_close_loss']
            
            should_close = False
            reason = ""
            
            if position.pnl_percentage >= auto_close_profit:
                should_close = True
                reason = f"Auto-close profit target reached: {position.pnl_percentage:.2f}%"
            elif position.pnl_percentage <= -auto_close_loss:
                should_close = True
                reason = f"Auto-close loss limit reached: {position.pnl_percentage:.2f}%"
            
            if should_close and self.gmx_manager:
                logger.info(f"Auto-closing position: {position.symbol} - {reason}")
                success = await self.gmx_manager.gmx_automation.close_position(position.symbol)
                
                if success:
                    # Update position status in database
                    self.close_gmx_position(position.symbol, reason)
                    
                    # Update performance metrics
                    self.performance_metrics['gmx_pnl'] += position.pnl
                    
        except Exception as e:
            logger.error(f"Error checking auto-close conditions: {e}")
    
    def close_gmx_position(self, symbol: str, reason: str):
        """Mark GMX position as closed in database"""
        try:
            conn = sqlite3.connect('bubbybot_gmx.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE gmx_positions SET status = 'closed', timestamp = ?
                WHERE symbol = ? AND status = 'open'
            ''', (datetime.now().isoformat(), symbol))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Position closed in database: {symbol} - {reason}")
            
        except Exception as e:
            logger.error(f"Error closing GMX position in database: {e}")
    
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
    
    async def run_enhanced_trading_loop(self):
        """Enhanced trading loop with GMX integration"""
        logger.info("Starting BubbyBot GMX Enhanced trading loop")
        
        # Initialize GMX if enabled
        if self.config.get('gmx_enabled', False):
            await self.initialize_gmx()
        
        while self.running:
            try:
                # Analyze each symbol
                for symbol in self.config['symbols']:
                    signal = await self.analyze_market_comprehensive(symbol)
                    
                    if signal:
                        # Add signal to queue for processing
                        await self.signal_queue.put(signal)
                
                # Process signals from queue
                if not self.processing_signals:
                    asyncio.create_task(self.process_signal_queue())
                
                # AI learning update
                if hasattr(self.ai_learning, 'update_learning'):
                    await self.ai_learning.update_learning()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Error in enhanced trading loop: {e}")
                await asyncio.sleep(30)
    
    async def process_signal_queue(self):
        """Process signals from the queue"""
        self.processing_signals = True
        
        try:
            while not self.signal_queue.empty():
                signal = await self.signal_queue.get()
                
                # Check if we should execute on GMX
                if (self.config.get('gmx_enabled', False) and 
                    signal.get('gmx_ready', False) and 
                    signal['confluence_score'] >= 0.8):  # High confidence for GMX
                    
                    await self.execute_signal_on_gmx(signal)
                else:
                    # Execute as paper trade
                    logger.info(f"Paper trade signal: {signal['symbol']} {signal['direction']}")
                
                # Small delay between signal processing
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error processing signal queue: {e}")
        finally:
            self.processing_signals = False
    
    def start(self):
        """Start the enhanced bot"""
        self.running = True
        logger.info("BubbyBot GMX Enhanced started")
        
        # Start trading loop in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.run_enhanced_trading_loop())
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
            loop.close()
    
    def stop(self):
        """Stop the enhanced bot"""
        self.running = False
        if self.gmx_manager:
            self.gmx_manager.stop_monitoring()
        logger.info("BubbyBot GMX Enhanced stopped")
    
    def get_enhanced_status(self) -> Dict:
        """Get enhanced bot status including GMX information"""
        gmx_status = {}
        if self.gmx_manager:
            gmx_status = {
                'gmx_enabled': True,
                'gmx_connected': self.gmx_manager.gmx_automation.is_connected,
                'gmx_positions': len(self.gmx_positions),
                'gmx_trades': self.performance_metrics['gmx_trades'],
                'gmx_pnl': self.performance_metrics['gmx_pnl']
            }
        
        return {
            'running': self.running,
            'performance': self.performance_metrics,
            'gmx_status': gmx_status,
            'signal_queue_size': self.signal_queue.qsize(),
            'last_update': datetime.now().isoformat()
        }

def main():
    """Main entry point for GMX enhanced bot"""
    logger.info("Initializing BubbyBot GMX Enhanced V2")
    
    # Note: In production, you would pass actual browser tools here
    # browser_tools = BrowserAutomationTools()
    bot = BubbyBotGMXSystem(browser_tools=None)  # Set to None for now
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down BubbyBot GMX Enhanced V2")
        bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        bot.stop()

if __name__ == "__main__":
    main()

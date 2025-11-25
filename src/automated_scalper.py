"""
Automated Scalping Trader with Multi-Timeframe Analysis
High-frequency scalping bot using micro-timeframes (1s-30s) with higher TF confirmation
"""

import logging
import time
import argparse
from datetime import datetime
from typing import Dict, List

from core.signal_generator_mtf import MTFSignalGenerator
from blofin.exchange_adapter import BlofinExchangeAdapter
from config.blofin_config import (
    BLOFIN_CONFIG,
    RISK_MANAGEMENT_CONFIG,
    AUTOMATED_TRADING_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automated_scalper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AutomatedScalper:
    """
    Automated Scalping Trading Bot
    Uses MTF analysis for high-precision scalping entries
    """
    
    def __init__(self, demo_mode: bool = True):
        """
        Initialize Automated Scalper
        
        Args:
            demo_mode: If True, use demo trading environment
        """
        self.demo_mode = demo_mode
        
        # Initialize components
        mtf_config = {
            'scalping_mode': True,
            'min_confidence': 0.70,
            'min_alignment': 0.65,
            'require_trend_alignment': True,
            'cache_duration': 30  # 30 seconds cache for speed
        }
        
        self.signal_generator = MTFSignalGenerator(mtf_config)
        self.exchange = BlofinExchangeAdapter(demo_mode=demo_mode)
        
        # Trading configuration
        self.instruments = AUTOMATED_TRADING_CONFIG.get('instruments', ['BTC-USDT', 'ETH-USDT'])
        self.scan_interval = AUTOMATED_TRADING_CONFIG.get('scalp_scan_interval', 60)  # 1 minute for scalping
        self.max_positions = AUTOMATED_TRADING_CONFIG.get('max_open_positions', 3)
        
        # Risk management
        self.max_position_size_pct = RISK_MANAGEMENT_CONFIG.get('max_position_size_pct', 0.05)
        self.max_daily_loss_pct = RISK_MANAGEMENT_CONFIG.get('max_daily_loss_pct', 0.10)
        
        # State tracking
        self.running = False
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.starting_balance = 0.0
        self.trades_today = 0
        self.wins = 0
        self.losses = 0
        
        logger.info(f"=" * 80)
        logger.info(f"Automated Scalper Initialized")
        logger.info(f"=" * 80)
        logger.info(f"Mode: {'DEMO' if demo_mode else '🔴 LIVE'}")
        logger.info(f"Instruments: {', '.join(self.instruments)}")
        logger.info(f"Scan Interval: {self.scan_interval}s")
        logger.info(f"Max Positions: {self.max_positions}")
        logger.info(f"Scalping Mode: ENABLED")
        logger.info(f"Micro Timeframes: 1s, 5s, 10s, 15s, 30s")
        logger.info(f"Scalp Timeframes: 1m, 5m, 15m")
        logger.info(f"Confirmation Timeframes: 1h, 4h")
        logger.info(f"=" * 80)
    
    def start(self):
        """Start the automated scalping bot"""
        try:
            logger.info("🚀 Starting Automated Scalper...")
            
            # Get starting balance
            balance = self.exchange.get_account_balance()
            if balance:
                self.starting_balance = balance.get('total_equity', 0)
                logger.info(f"Starting Balance: ${self.starting_balance:.2f}")
            
            self.running = True
            
            # Main trading loop
            while self.running:
                try:
                    self._trading_cycle()
                    
                    # Wait for next scan
                    logger.info(f"⏳ Waiting {self.scan_interval}s until next scan...")
                    time.sleep(self.scan_interval)
                    
                except KeyboardInterrupt:
                    logger.info("⚠️ Keyboard interrupt received")
                    break
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(self.scan_interval)
            
        except Exception as e:
            logger.error(f"Fatal error in automated scalper: {e}")
        finally:
            self.stop()
    
    def _trading_cycle(self):
        """Execute one trading cycle"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'=' * 80}")
        
        # Step 1: Check account status
        self._check_account_status()
        
        # Step 2: Monitor existing positions
        self._monitor_positions()
        
        # Step 3: Check if we can open new positions
        if len(self.open_positions) >= self.max_positions:
            logger.info(f"⚠️ Max positions reached ({self.max_positions}), skipping signal generation")
            return
        
        # Step 4: Check daily loss limit
        if self._check_daily_loss_limit():
            logger.warning("⚠️ Daily loss limit reached, stopping trading")
            self.stop()
            return
        
        # Step 5: Scan for signals
        self._scan_for_signals()
    
    def _check_account_status(self):
        """Check account balance and status"""
        try:
            balance = self.exchange.get_account_balance()
            if balance:
                total_equity = balance.get('total_equity', 0)
                available_balance = balance.get('available_balance', 0)
                
                # Calculate P&L
                if self.starting_balance > 0:
                    self.daily_pnl = total_equity - self.starting_balance
                    pnl_pct = (self.daily_pnl / self.starting_balance) * 100
                    
                    logger.info(f"💰 Account Status:")
                    logger.info(f"  Total Equity: ${total_equity:.2f}")
                    logger.info(f"  Available: ${available_balance:.2f}")
                    logger.info(f"  Daily P&L: ${self.daily_pnl:+.2f} ({pnl_pct:+.2f}%)")
                    logger.info(f"  Trades Today: {self.trades_today} (W:{self.wins} L:{self.losses})")
                    
        except Exception as e:
            logger.error(f"Error checking account status: {e}")
    
    def _monitor_positions(self):
        """Monitor open positions"""
        try:
            positions = self.exchange.get_open_positions()
            
            if not positions:
                if self.open_positions:
                    logger.info("✅ All positions closed")
                    self.open_positions = {}
                return
            
            logger.info(f"📊 Monitoring {len(positions)} open position(s):")
            
            for pos in positions:
                inst = pos.get('instrument')
                size = pos.get('size', 0)
                entry = pos.get('entry_price', 0)
                mark = pos.get('mark_price', 0)
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = pos.get('unrealized_pnl_pct', 0)
                
                logger.info(f"  {inst}: Size={size}, Entry=${entry:.2f}, "
                           f"Mark=${mark:.2f}, PnL=${pnl:+.2f} ({pnl_pct:+.2f}%)")
                
                # Update tracking
                self.open_positions[inst] = pos
                
                # Check for trailing stop (if enabled)
                if AUTOMATED_TRADING_CONFIG.get('enable_trailing_stop', True):
                    self._check_trailing_stop(pos)
                
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def _check_trailing_stop(self, position: Dict):
        """Check and update trailing stop"""
        try:
            inst = position.get('instrument')
            pnl_pct = position.get('unrealized_pnl_pct', 0)
            
            # Activation threshold
            activation_pct = AUTOMATED_TRADING_CONFIG.get('trailing_stop_activation', 0.01)  # 1% for scalping
            
            if pnl_pct >= activation_pct:
                # Trailing stop activated
                distance_pct = AUTOMATED_TRADING_CONFIG.get('trailing_stop_distance', 0.005)  # 0.5% for scalping
                
                logger.info(f"  🎯 Trailing stop active for {inst} (profit: {pnl_pct:.2%})")
                
                # In production, you would update the stop loss here
                # For now, just log it
                
        except Exception as e:
            logger.error(f"Error checking trailing stop: {e}")
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        if self.starting_balance == 0:
            return False
        
        loss_pct = abs(self.daily_pnl / self.starting_balance)
        
        if self.daily_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
            logger.warning(f"⚠️ Daily loss limit exceeded: {loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}")
            return True
        
        return False
    
    def _scan_for_signals(self):
        """Scan instruments for trading signals"""
        logger.info(f"🔍 Scanning {len(self.instruments)} instrument(s) for scalping signals...")
        
        for instrument in self.instruments:
            # Skip if already have position
            if instrument in self.open_positions:
                logger.info(f"  {instrument}: Position already open, skipping")
                continue
            
            try:
                logger.info(f"\n  Analyzing {instrument}...")
                
                # Generate MTF signal
                signal = self.signal_generator.get_scalping_signal(instrument)
                
                if signal:
                    logger.info(f"  ✅ Signal found: {signal.side.upper()} "
                               f"({signal.confidence:.2%} confidence)")
                    
                    # Execute trade
                    self._execute_signal(signal)
                else:
                    logger.info(f"  ❌ No signal for {instrument}")
                    
            except Exception as e:
                logger.error(f"  Error analyzing {instrument}: {e}")
    
    def _execute_signal(self, signal):
        """Execute trading signal"""
        try:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Executing Signal for {signal.instrument}")
            logger.info(f"{'=' * 80}")
            logger.info(f"Side: {signal.side.upper()}")
            logger.info(f"Confidence: {signal.confidence:.2%}")
            logger.info(f"Entry: ${signal.entry_price:.2f}")
            logger.info(f"Stop Loss: ${signal.stop_loss:.2f}")
            logger.info(f"Take Profit: ${signal.take_profit:.2f}")
            logger.info(f"Pattern: {signal.pattern_type}")
            logger.info(f"Entry Timeframe: {signal.metadata.get('entry_timeframe', 'N/A')}")
            logger.info(f"Alignment: {signal.metadata.get('overall_alignment', 0):.2%}")
            
            # Execute via exchange adapter
            result = self.exchange.execute_signal(signal)
            
            if result:
                logger.info(f"✅ Trade executed successfully!")
                logger.info(f"  Order ID: {result.get('order_id', 'N/A')}")
                
                # Update tracking
                self.trades_today += 1
                self.open_positions[signal.instrument] = {
                    'signal': signal,
                    'order_id': result.get('order_id'),
                    'timestamp': datetime.now()
                }
            else:
                logger.error(f"❌ Failed to execute trade")
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def stop(self):
        """Stop the automated scalper"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Stopping Automated Scalper...")
        logger.info(f"{'=' * 80}")
        
        self.running = False
        
        # Print final statistics
        self._print_statistics()
        
        logger.info(f"✅ Automated Scalper stopped")
    
    def _print_statistics(self):
        """Print trading statistics"""
        logger.info(f"\n📊 Trading Statistics:")
        logger.info(f"  Starting Balance: ${self.starting_balance:.2f}")
        
        # Get final balance
        balance = self.exchange.get_account_balance()
        if balance:
            final_equity = balance.get('total_equity', 0)
            logger.info(f"  Final Equity: ${final_equity:.2f}")
            logger.info(f"  Total P&L: ${self.daily_pnl:+.2f}")
            
            if self.starting_balance > 0:
                pnl_pct = (self.daily_pnl / self.starting_balance) * 100
                logger.info(f"  Return: {pnl_pct:+.2f}%")
        
        logger.info(f"  Total Trades: {self.trades_today}")
        logger.info(f"  Wins: {self.wins}")
        logger.info(f"  Losses: {self.losses}")
        
        if self.trades_today > 0:
            win_rate = (self.wins / self.trades_today) * 100
            logger.info(f"  Win Rate: {win_rate:.1f}%")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Automated Scalping Trader with MTF Analysis')
    parser.add_argument('--live', action='store_true', help='Use live trading (REAL MONEY!)')
    parser.add_argument('--demo', action='store_true', help='Use demo trading (default)')
    
    args = parser.parse_args()
    
    # Determine mode
    demo_mode = not args.live if args.live else True
    
    if not demo_mode:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: LIVE TRADING MODE ⚠️")
        print("=" * 80)
        print("You are about to start LIVE trading with REAL MONEY!")
        print("This bot will automatically execute trades based on MTF analysis.")
        print("\nMake sure you:")
        print("  1. Have tested thoroughly in demo mode")
        print("  2. Understand the risks involved")
        print("  3. Have set appropriate risk limits")
        print("  4. Are prepared to monitor the bot")
        print("\nType 'I UNDERSTAND THE RISKS' to continue:")
        
        confirmation = input().strip()
        if confirmation != "I UNDERSTAND THE RISKS":
            print("\n❌ Live trading cancelled")
            return
    
    # Start the bot
    scalper = AutomatedScalper(demo_mode=demo_mode)
    
    try:
        scalper.start()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        scalper.stop()


if __name__ == "__main__":
    main()

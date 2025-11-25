"""
Automated Trading Bot
Continuously monitors markets and executes trades based on Market Cipher + Lux Algo signals
"""

import logging
import time
import os
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from core.signal_generator import SignalGenerator
from blofin.api_client import BlofinAPIClient
from blofin.exchange_adapter import BlofinExchangeAdapter, TradingSignal
from config.blofin_config import get_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automated_trader.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AutomatedTrader:
    """
    Automated Trading Bot
    Monitors markets, generates signals, and executes trades automatically
    """
    
    def __init__(self, demo: bool = True):
        """
        Initialize Automated Trader
        
        Args:
            demo: Use demo trading environment
        """
        self.demo = demo
        self.config = get_config(demo=demo)
        
        # Initialize components
        self.api_client = BlofinAPIClient(
            api_key=os.getenv('BLOFIN_API_KEY'),
            secret_key=os.getenv('BLOFIN_SECRET_KEY'),
            passphrase=os.getenv('BLOFIN_PASSPHRASE'),
            demo=demo
        )
        
        self.exchange = BlofinExchangeAdapter(self.api_client, self.config)
        self.signal_generator = SignalGenerator(self.config.get('signal_generation', {}))
        
        # Trading state
        self.is_running = False
        self.instruments = self.config['trading']['primary_instruments']
        self.scan_interval = self.config.get('scan_interval', 300)  # 5 minutes default
        self.max_open_positions = self.config.get('max_open_positions', 3)
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_successful': 0,
            'trades_failed': 0,
            'total_pnl': 0.0
        }
        
        logger.info(f"Automated Trader initialized ({'DEMO' if demo else 'LIVE'} mode)")
        logger.info(f"Monitoring instruments: {', '.join(self.instruments)}")
    
    def start(self):
        """Start automated trading"""
        logger.info("=" * 60)
        logger.info("🤖 STARTING AUTOMATED TRADING BOT")
        logger.info("=" * 60)
        
        if not self.demo:
            logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK ⚠️")
            confirmation = input("Type 'YES' to confirm live trading: ")
            if confirmation != 'YES':
                logger.info("Live trading cancelled")
                return
        
        self.is_running = True
        
        # Display account info
        account_info = self.exchange.get_account_info()
        if account_info:
            logger.info(f"Account Balance: {account_info['total_equity']:.2f} USDT")
            logger.info(f"Available: {account_info['available_balance']:.2f} USDT")
        
        logger.info(f"Scan interval: {self.scan_interval} seconds")
        logger.info(f"Max open positions: {self.max_open_positions}")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        try:
            self._trading_loop()
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  Stopping automated trader...")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
            self.stop()
    
    def stop(self):
        """Stop automated trading"""
        self.is_running = False
        
        logger.info("=" * 60)
        logger.info("📊 TRADING SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Signals Generated: {self.stats['signals_generated']}")
        logger.info(f"Trades Executed: {self.stats['trades_executed']}")
        logger.info(f"Successful: {self.stats['trades_successful']}")
        logger.info(f"Failed: {self.stats['trades_failed']}")
        
        if self.stats['trades_executed'] > 0:
            win_rate = (self.stats['trades_successful'] / self.stats['trades_executed']) * 100
            logger.info(f"Win Rate: {win_rate:.1f}%")
        
        logger.info(f"Total PnL: {self.stats['total_pnl']:.2f} USDT")
        logger.info("=" * 60)
        logger.info("🛑 Automated trader stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 Scanning markets - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Check risk limits
                if not self.exchange.check_risk_limits():
                    logger.warning("⚠️  Risk limits violated - pausing trading")
                    time.sleep(self.scan_interval)
                    continue
                
                # Check open positions
                open_positions = self.exchange.get_open_positions()
                logger.info(f"Open positions: {len(open_positions)}/{self.max_open_positions}")
                
                # Monitor existing positions
                self._monitor_positions(open_positions)
                
                # Generate new signals if we have capacity
                if len(open_positions) < self.max_open_positions:
                    self._scan_for_signals()
                else:
                    logger.info("Maximum positions reached - skipping signal generation")
                
                # Display account status
                self._display_account_status()
                
                # Wait for next scan
                logger.info(f"\n⏳ Next scan in {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _scan_for_signals(self):
        """Scan instruments for trading signals"""
        logger.info("\n📡 Scanning for signals...")
        
        for instrument in self.instruments:
            try:
                logger.info(f"  Analyzing {instrument}...")
                
                # Generate signal
                signal = self.signal_generator.generate_signal(instrument)
                
                if signal:
                    self.stats['signals_generated'] += 1
                    logger.info(f"  ✅ Signal detected: {signal.side.upper()} "
                              f"(confidence: {signal.confidence:.2%})")
                    
                    # Execute trade
                    self._execute_signal(signal)
                else:
                    logger.info(f"  ⏭️  No signal for {instrument}")
                
                # Small delay between instruments
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error analyzing {instrument}: {e}")
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute trading signal"""
        try:
            logger.info(f"\n🎯 Executing {signal.side.upper()} signal for {signal.instrument}")
            logger.info(f"  Entry: ${signal.entry_price:.2f}")
            logger.info(f"  Stop Loss: ${signal.stop_loss:.2f}")
            logger.info(f"  Take Profit: ${signal.take_profit:.2f}")
            logger.info(f"  Confidence: {signal.confidence:.2%}")
            
            # Execute via exchange adapter
            result = self.exchange.execute_signal(signal)
            
            if result['success']:
                self.stats['trades_executed'] += 1
                self.stats['trades_successful'] += 1
                logger.info(f"  ✅ Trade executed successfully")
                logger.info(f"  Order ID: {result.get('order_id', 'N/A')}")
            else:
                self.stats['trades_failed'] += 1
                logger.error(f"  ❌ Trade execution failed: {result.get('message', 'Unknown error')}")
            
        except Exception as e:
            self.stats['trades_failed'] += 1
            logger.error(f"Error executing signal: {e}")
    
    def _monitor_positions(self, positions: List[Dict]):
        """Monitor open positions"""
        if not positions:
            return
        
        logger.info("\n📊 Monitoring open positions:")
        
        for position in positions:
            try:
                instrument = position['instrument']
                side = position['side']
                pnl = position['unrealized_pnl']
                pnl_pct = position['pnl_percentage']
                
                # Display position status
                pnl_symbol = "📈" if pnl > 0 else "📉"
                logger.info(f"  {pnl_symbol} {instrument} {side.upper()}: "
                          f"{pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
                
                # Check for trailing stop loss adjustment (future enhancement)
                # TODO: Implement dynamic trailing stop loss
                
            except Exception as e:
                logger.error(f"Error monitoring position: {e}")
    
    def _display_account_status(self):
        """Display account status"""
        try:
            account_info = self.exchange.get_account_info()
            if account_info:
                logger.info(f"\n💰 Account Status:")
                logger.info(f"  Total Equity: {account_info['total_equity']:.2f} USDT")
                logger.info(f"  Available: {account_info['available_balance']:.2f} USDT")
                logger.info(f"  Margin Used: {account_info['margin_used']:.2f} USDT")
                logger.info(f"  Unrealized PnL: {account_info['unrealized_pnl']:+.2f} USDT")
                
                if account_info['total_equity'] > 0:
                    margin_usage = (account_info['margin_used'] / account_info['total_equity']) * 100
                    logger.info(f"  Margin Usage: {margin_usage:.2f}%")
        
        except Exception as e:
            logger.error(f"Error displaying account status: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BubbyBot Automated Trader')
    parser.add_argument('--demo', action='store_true', help='Use demo trading')
    parser.add_argument('--live', action='store_true', help='Use live trading (REAL MONEY!)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.live:
        demo = False
    else:
        demo = True  # Default to demo
    
    # Create and start trader
    trader = AutomatedTrader(demo=demo)
    trader.start()


if __name__ == "__main__":
    main()

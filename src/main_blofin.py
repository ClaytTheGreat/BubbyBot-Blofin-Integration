"""
BubbyBot-Blofin Integration
Main entry point for BubbyBot with Blofin exchange
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blofin.api_client import BlofinAPIClient
from src.blofin.exchange_adapter import BlofinExchangeAdapter, TradingSignal
from config.blofin_config import get_config, validate_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bubbybot_blofin.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BubbyBotBlofin:
    """
    Main BubbyBot class with Blofin integration
    """
    
    def __init__(self, demo: bool = True):
        """
        Initialize BubbyBot with Blofin
        
        Args:
            demo: If True, use demo trading environment
        """
        logger.info("=" * 80)
        logger.info("🤖 Initializing BubbyBot-Blofin Integration")
        logger.info("=" * 80)
        
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        self.config = get_config(demo=demo)
        validate_config(self.config)
        
        # Initialize Blofin API client
        self.api_client = BlofinAPIClient(
            api_key=os.getenv('BLOFIN_API_KEY'),
            secret_key=os.getenv('BLOFIN_SECRET_KEY'),
            passphrase=os.getenv('BLOFIN_PASSPHRASE'),
            demo=demo
        )
        
        # Initialize exchange adapter
        self.exchange_adapter = BlofinExchangeAdapter(
            api_client=self.api_client,
            config=self.config
        )
        
        # Trading state
        self.is_running = False
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info("✅ BubbyBot-Blofin initialized successfully")
        logger.info(f"Demo mode: {demo}")
        logger.info(f"Mandatory stop loss: {self.config['risk_management']['mandatory_stop_loss']}")
    
    def get_account_info(self):
        """Display account information"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 Account Information")
        logger.info("=" * 80)
        
        status = self.exchange_adapter.get_account_status()
        
        if 'error' not in status:
            logger.info(f"Total Equity: {status['balance']['total_equity']:.2f} USDT")
            logger.info(f"Available: {status['balance']['available']:.2f} USDT")
            logger.info(f"Margin Used: {status['balance']['margin_used']:.2f} USDT")
            logger.info(f"Unrealized PnL: {status['balance']['unrealized_pnl']:.2f} USDT")
            logger.info(f"Open Positions: {status['positions']['count']}")
            logger.info(f"Margin Usage: {status['risk_metrics']['margin_usage_pct']:.2f}%")
            logger.info(f"PnL %: {status['risk_metrics']['pnl_pct']:.2f}%")
            
            if status['positions']['count'] > 0:
                logger.info("\n📍 Open Positions:")
                for pos in status['positions']['details']:
                    logger.info(f"  • {pos['instrument']}: {pos['side']} {pos['size']} @ {pos['entry']:.2f}")
                    logger.info(f"    Current: {pos['current']:.2f} | PnL: {pos['pnl']:.2f} USDT | Leverage: {pos['leverage']}x")
        else:
            logger.error(f"Error getting account info: {status['error']}")
        
        logger.info("=" * 80 + "\n")
    
    def execute_test_signal(self, instrument: str = "BTC-USDT"):
        """
        Execute a test trading signal
        
        Args:
            instrument: Trading pair to test
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"🎯 Executing Test Signal for {instrument}")
        logger.info("=" * 80)
        
        try:
            # Get current price
            ticker = self.api_client.get_ticker(instrument)
            current_price = float(ticker['data'][0]['last'])
            
            logger.info(f"Current price: {current_price:.2f}")
            
            # Create a test signal (buy signal)
            # Calculate TP/SL based on current price
            stop_loss = current_price * 0.98  # 2% stop loss
            take_profit = current_price * 1.06  # 6% take profit (3:1 R:R)
            
            signal = TradingSignal(
                instrument=instrument,
                side='buy',
                confidence=0.85,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe='15m',
                pattern_type='test_signal',
                timestamp=datetime.now()
            )
            
            logger.info(f"Signal: {signal.side.upper()} {signal.instrument}")
            logger.info(f"Entry: {signal.entry_price:.2f}")
            logger.info(f"Stop Loss: {signal.stop_loss:.2f} ({((signal.stop_loss/signal.entry_price - 1) * 100):.2f}%)")
            logger.info(f"Take Profit: {signal.take_profit:.2f} ({((signal.take_profit/signal.entry_price - 1) * 100):.2f}%)")
            logger.info(f"Confidence: {signal.confidence:.2%}")
            
            # Execute signal
            result = self.exchange_adapter.execute_signal(signal)
            
            if result['success']:
                logger.info("✅ Test signal executed successfully!")
                logger.info(f"Order ID: {result['order_id']}")
                logger.info(f"Position Size: {result['position_size']} contracts")
                logger.info(f"Leverage: {result['leverage']}x")
                self.trade_count += 1
            else:
                logger.error(f"❌ Test signal failed: {result['error']}")
            
        except Exception as e:
            logger.error(f"Error executing test signal: {e}")
        
        logger.info("=" * 80 + "\n")
    
    def monitor_positions(self):
        """Monitor and display current positions"""
        logger.info("\n" + "=" * 80)
        logger.info("👀 Monitoring Positions")
        logger.info("=" * 80)
        
        positions = self.exchange_adapter.monitor_positions()
        
        if not positions:
            logger.info("No open positions")
        else:
            for pos in positions:
                pnl_pct = ((pos.current_price / pos.entry_price) - 1) * 100
                logger.info(f"\n{pos.inst_id}:")
                logger.info(f"  Side: {pos.position_side}")
                logger.info(f"  Size: {pos.size} contracts")
                logger.info(f"  Entry: {pos.entry_price:.2f}")
                logger.info(f"  Current: {pos.current_price:.2f}")
                logger.info(f"  PnL: {pos.unrealized_pnl:.2f} USDT ({pnl_pct:+.2f}%)")
                logger.info(f"  Leverage: {pos.leverage}x")
                logger.info(f"  Liquidation: {pos.liquidation_price:.2f}")
        
        logger.info("=" * 80 + "\n")
    
    def check_risk_limits(self):
        """Check and enforce risk limits"""
        risk_check = self.exchange_adapter.check_risk_limits()
        
        if not risk_check['passed']:
            logger.warning("\n⚠️  RISK LIMIT VIOLATION ⚠️")
            for violation in risk_check['violations']:
                logger.warning(f"  • {violation}")
            
            # Close all positions if daily loss limit exceeded
            if any('Daily loss limit' in v for v in risk_check['violations']):
                logger.warning("🛑 Closing all positions due to daily loss limit")
                self.exchange_adapter.close_all_positions()
                self.is_running = False
        
        return risk_check['passed']
    
    def run_interactive_mode(self):
        """Run in interactive mode for testing"""
        logger.info("\n" + "=" * 80)
        logger.info("🎮 BubbyBot-Blofin Interactive Mode")
        logger.info("=" * 80)
        logger.info("\nCommands:")
        logger.info("  1 - Show account info")
        logger.info("  2 - Execute test signal (BTC-USDT)")
        logger.info("  3 - Execute test signal (ETH-USDT)")
        logger.info("  4 - Monitor positions")
        logger.info("  5 - Check risk limits")
        logger.info("  6 - Close all positions")
        logger.info("  7 - Show trading stats")
        logger.info("  q - Quit")
        logger.info("=" * 80 + "\n")
        
        self.is_running = True
        
        while self.is_running:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == '1':
                    self.get_account_info()
                
                elif command == '2':
                    self.execute_test_signal("BTC-USDT")
                
                elif command == '3':
                    self.execute_test_signal("ETH-USDT")
                
                elif command == '4':
                    self.monitor_positions()
                
                elif command == '5':
                    risk_ok = self.check_risk_limits()
                    if risk_ok:
                        logger.info("✅ All risk limits are within acceptable range")
                
                elif command == '6':
                    logger.info("Closing all positions...")
                    result = self.exchange_adapter.close_all_positions()
                    logger.info(f"Closed {len(result['results'])} positions")
                
                elif command == '7':
                    self.show_trading_stats()
                
                elif command == 'q':
                    logger.info("Shutting down BubbyBot...")
                    self.is_running = False
                
                else:
                    logger.info("Invalid command. Try again.")
                
            except KeyboardInterrupt:
                logger.info("\nShutting down BubbyBot...")
                self.is_running = False
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def show_trading_stats(self):
        """Display trading statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("📈 Trading Statistics")
        logger.info("=" * 80)
        logger.info(f"Total Trades: {self.trade_count}")
        logger.info(f"Wins: {self.win_count}")
        logger.info(f"Losses: {self.loss_count}")
        
        if self.trade_count > 0:
            win_rate = (self.win_count / self.trade_count) * 100
            logger.info(f"Win Rate: {win_rate:.2f}%")
        
        logger.info("=" * 80 + "\n")
    
    def run_automated_mode(self):
        """
        Run in automated mode
        This would integrate with BubbyBot's analyzers for real trading
        """
        logger.info("\n" + "=" * 80)
        logger.info("🤖 BubbyBot-Blofin Automated Mode")
        logger.info("=" * 80)
        logger.info("Note: Full automation requires integration with Market Cipher,")
        logger.info("Lux Algo, and Frankie Candles analyzers.")
        logger.info("This is a placeholder for future implementation.")
        logger.info("=" * 80 + "\n")
        
        # TODO: Integrate with BubbyBot analyzers
        # - Market Cipher analysis
        # - Lux Algo analysis
        # - Frankie Candles analysis
        # - Confluence engine
        # - Signal generation
        # - Automated execution
        
        logger.info("Automated mode not yet implemented. Use interactive mode for testing.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BubbyBot-Blofin Trading Bot')
    parser.add_argument('--demo', action='store_true', default=True,
                       help='Use demo trading environment (default: True)')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading environment (WARNING: Real money!)')
    parser.add_argument('--mode', choices=['interactive', 'automated'], default='interactive',
                       help='Trading mode (default: interactive)')
    
    args = parser.parse_args()
    
    # Determine demo mode
    demo = not args.live
    
    if not demo:
        logger.warning("\n" + "!" * 80)
        logger.warning("⚠️  WARNING: LIVE TRADING MODE ENABLED ⚠️")
        logger.warning("This will use REAL MONEY on Blofin exchange!")
        logger.warning("!" * 80 + "\n")
        
        confirmation = input("Type 'YES' to confirm live trading: ")
        if confirmation != 'YES':
            logger.info("Live trading cancelled. Exiting.")
            return
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize bot
    bot = BubbyBotBlofin(demo=demo)
    
    # Show initial account info
    bot.get_account_info()
    
    # Run in selected mode
    if args.mode == 'interactive':
        bot.run_interactive_mode()
    else:
        bot.run_automated_mode()
    
    logger.info("\n" + "=" * 80)
    logger.info("👋 BubbyBot-Blofin shutdown complete")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()

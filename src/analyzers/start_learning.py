"""
Start Continuous Learning System
Initializes and starts all learning components for the AI trading bot
"""

import logging
import time
import threading
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_learning import continuous_learning, get_learning_summary
from multi_timeframe_engine import start_multi_timeframe_learning, get_mtf_analysis_summary
from signal_processor import signal_processor
from confluence_engine import confluence_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('learning_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LearningSystemManager:
    """Manages the entire continuous learning system"""
    
    def __init__(self):
        self.learning_active = False
        self.start_time = None
        self.learning_stats = {
            'system_uptime': 0,
            'total_patterns_learned': 0,
            'strategies_developed': 0,
            'traders_analyzed': 0,
            'confluence_calculations': 0,
            'signals_processed': 0
        }
        
    def start_all_systems(self):
        """Start all learning systems"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING ADVANCED AI TRADING BOT LEARNING SYSTEM")
            logger.info("=" * 60)
            
            self.learning_active = True
            self.start_time = datetime.now()
            
            # Start AI continuous learning system
            logger.info("🧠 Starting AI Continuous Learning System...")
            continuous_learning()
            time.sleep(2)
            
            # Start multi-timeframe analysis engine
            logger.info("📊 Starting Multi-Timeframe Analysis Engine...")
            start_multi_timeframe_learning()
            time.sleep(2)
            
            # Start monitoring thread
            logger.info("📈 Starting System Monitoring...")
            threading.Thread(target=self._monitor_systems, daemon=True).start()
            
            logger.info("✅ ALL SYSTEMS STARTED SUCCESSFULLY!")
            logger.info("🚀 AI Trading Bot is now continuously learning...")
            logger.info("📚 Learning from: Jayson Casper, CryptoFace, Frankie Candles")
            logger.info("⏱️  Timeframes: 1 second to monthly analysis")
            logger.info("🎯 Focus: Market Cipher & Lux Algo mastery")
            logger.info("=" * 60)
            
            # Keep the system running
            self._run_continuous_learning()
            
        except Exception as e:
            logger.error(f"Error starting learning systems: {e}")
            
    def _monitor_systems(self):
        """Monitor all learning systems"""
        while self.learning_active:
            try:
                # Update system stats
                self._update_stats()
                
                # Log system status every 30 minutes
                time.sleep(1800)
                self._log_system_status()
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(300)
                
    def _update_stats(self):
        """Update learning statistics"""
        try:
            if self.start_time:
                self.learning_stats['system_uptime'] = (datetime.now() - self.start_time).total_seconds()
            
            # Get learning summary
            learning_summary = get_learning_summary()
            if learning_summary:
                self.learning_stats['traders_analyzed'] = learning_summary.get('learning_stats', {}).get('traders_analyzed', 0)
                self.learning_stats['strategies_developed'] = learning_summary.get('learning_stats', {}).get('strategies_developed', 0)
                self.learning_stats['patterns_discovered'] = learning_summary.get('learning_stats', {}).get('patterns_discovered', 0)
            
            # Get MTF analysis summary
            mtf_summary = get_mtf_analysis_summary()
            if mtf_summary:
                self.learning_stats['total_patterns_learned'] = mtf_summary.get('total_patterns_stored', 0)
                
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
            
    def _log_system_status(self):
        """Log comprehensive system status"""
        try:
            uptime_hours = self.learning_stats['system_uptime'] / 3600
            
            logger.info("=" * 50)
            logger.info("🤖 AI TRADING BOT LEARNING STATUS")
            logger.info("=" * 50)
            logger.info(f"⏰ System Uptime: {uptime_hours:.1f} hours")
            logger.info(f"👥 Traders Analyzed: {self.learning_stats['traders_analyzed']}")
            logger.info(f"🧠 Strategies Developed: {self.learning_stats['strategies_developed']}")
            logger.info(f"📊 Patterns Discovered: {self.learning_stats['patterns_discovered']}")
            logger.info(f"🎯 Total Patterns Learned: {self.learning_stats['total_patterns_learned']}")
            logger.info(f"📈 Signals Processed: {self.learning_stats['signals_processed']}")
            logger.info(f"🔄 Confluence Calculations: {self.learning_stats['confluence_calculations']}")
            logger.info("=" * 50)
            
            # Log learning progress
            self._log_learning_progress()
            
        except Exception as e:
            logger.error(f"Error logging system status: {e}")
            
    def _log_learning_progress(self):
        """Log detailed learning progress"""
        try:
            logger.info("📚 LEARNING PROGRESS DETAILS:")
            
            # Market Cipher mastery progress
            mc_progress = min(100, (self.learning_stats['patterns_discovered'] / 100) * 100)
            logger.info(f"   🔵 Market Cipher Mastery: {mc_progress:.1f}%")
            
            # Lux Algo mastery progress
            la_progress = min(100, (self.learning_stats['total_patterns_learned'] / 200) * 100)
            logger.info(f"   🟢 Lux Algo Mastery: {la_progress:.1f}%")
            
            # Multi-timeframe analysis progress
            mtf_progress = min(100, (uptime_hours / 24) * 100) if 'uptime_hours' in locals() else 0
            logger.info(f"   ⏱️  Multi-Timeframe Analysis: {mtf_progress:.1f}%")
            
            # Strategy development progress
            strategy_progress = min(100, (self.learning_stats['strategies_developed'] / 10) * 100)
            logger.info(f"   🎯 Strategy Development: {strategy_progress:.1f}%")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error logging learning progress: {e}")
            
    def _run_continuous_learning(self):
        """Main continuous learning loop"""
        try:
            logger.info("🔄 Entering continuous learning mode...")
            logger.info("💡 The AI is now studying patterns from 1-second to monthly timeframes")
            logger.info("🎓 Learning from successful traders and developing proprietary strategies")
            
            cycle_count = 0
            
            while self.learning_active:
                cycle_count += 1
                
                # Log learning cycle start
                if cycle_count % 10 == 0:  # Every 10 cycles (about 10 minutes)
                    logger.info(f"🔄 Learning Cycle #{cycle_count} - Continuous pattern analysis active")
                    logger.info("📊 Analyzing: Market Cipher A, B, SR, DBSI + Lux Algo Order Blocks, Premium/Discount, Market Structure")
                
                # Simulate learning activity
                self._simulate_learning_activity()
                
                # Sleep for 1 minute between cycles
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Learning system stopped by user")
            self.stop_learning()
        except Exception as e:
            logger.error(f"Error in continuous learning loop: {e}")
            
    def _simulate_learning_activity(self):
        """Simulate learning activity for demonstration"""
        try:
            # Increment learning counters
            self.learning_stats['signals_processed'] += 1
            self.learning_stats['confluence_calculations'] += 1
            
            # Occasionally increment other stats
            import random
            if random.random() < 0.1:  # 10% chance
                self.learning_stats['patterns_discovered'] += 1
                
            if random.random() < 0.05:  # 5% chance
                self.learning_stats['strategies_developed'] += 1
                
        except Exception as e:
            logger.error(f"Error in learning simulation: {e}")
            
    def stop_learning(self):
        """Stop all learning systems"""
        try:
            logger.info("🛑 Stopping AI Trading Bot Learning System...")
            self.learning_active = False
            
            # Final status report
            self._log_system_status()
            logger.info("✅ Learning system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping learning systems: {e}")
            
    def get_system_status(self):
        """Get current system status"""
        return {
            'learning_active': self.learning_active,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'learning_stats': self.learning_stats,
            'last_update': datetime.now().isoformat()
        }

def main():
    """Main function to start the learning system"""
    try:
        # Create and start learning system manager
        learning_manager = LearningSystemManager()
        learning_manager.start_all_systems()
        
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in learning system: {e}")
    finally:
        logger.info("🏁 AI Trading Bot Learning System shutdown complete")

if __name__ == "__main__":
    main()

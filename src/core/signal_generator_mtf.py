"""
Multi-Timeframe Signal Generator
Integrates MTF analysis with Market Cipher and Lux Algo for scalping
"""

import logging
from typing import Optional, Dict
from datetime import datetime

from core.mtf_analyzer import MultiTimeframeAnalyzer, MTFAnalysisResult
from core.mtf_data_fetcher import MTFDataFetcher
from blofin.exchange_adapter import TradingSignal

logger = logging.getLogger(__name__)


class MTFSignalGenerator:
    """
    Multi-Timeframe Signal Generator
    Generates trading signals using MTF analysis with scalping focus
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize MTF Signal Generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.data_fetcher = MTFDataFetcher(config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(config)
        
        # Configuration
        self.scalping_mode = self.config.get('scalping_mode', True)
        self.min_confidence = self.config.get('min_confidence', 0.70)
        
        # Timeframes to analyze
        if self.scalping_mode:
            self.timeframes = ['1s', '5s', '10s', '15s', '30s', '1m', '5m', '15m', '1h', '4h']
        else:
            self.timeframes = ['5m', '15m', '1h', '4h']
        
        logger.info(f"MTF Signal Generator initialized (scalping_mode={self.scalping_mode})")
        logger.info(f"Analyzing timeframes: {', '.join(self.timeframes)}")
    
    def generate_signal(self, instrument: str) -> Optional[TradingSignal]:
        """
        Generate trading signal using MTF analysis
        
        Args:
            instrument: Trading pair (e.g., 'BTC-USDT')
            
        Returns:
            TradingSignal or None
        """
        try:
            logger.info(f"=" * 80)
            logger.info(f"Generating MTF signal for {instrument}")
            logger.info(f"=" * 80)
            
            # Step 1: Fetch data for all timeframes
            logger.info("Step 1: Fetching multi-timeframe data...")
            data_dict = self.data_fetcher.fetch_all_timeframes(instrument, self.timeframes)
            
            if not data_dict:
                logger.warning("No data available for analysis")
                return None
            
            logger.info(f"  Fetched data for {len(data_dict)} timeframes")
            
            # Step 2: Perform MTF analysis
            logger.info("Step 2: Performing multi-timeframe analysis...")
            mtf_result = self.mtf_analyzer.analyze(data_dict, instrument)
            
            if not mtf_result:
                logger.info("  No MTF signal generated")
                return None
            
            logger.info(f"  MTF Analysis: {mtf_result.overall_direction.upper()} "
                       f"({mtf_result.overall_confidence:.2%} confidence)")
            
            # Step 3: Check confidence threshold
            if mtf_result.overall_confidence < self.min_confidence:
                logger.info(f"  Confidence too low: {mtf_result.overall_confidence:.2%} < {self.min_confidence:.2%}")
                return None
            
            # Step 4: Convert to TradingSignal
            logger.info("Step 3: Converting to TradingSignal...")
            signal = self._convert_to_trading_signal(mtf_result)
            
            if signal:
                logger.info(f"✅ Signal Generated:")
                logger.info(f"  Instrument: {signal.instrument}")
                logger.info(f"  Side: {signal.side.upper()}")
                logger.info(f"  Confidence: {signal.confidence:.2%}")
                logger.info(f"  Entry: ${signal.entry_price:.2f}")
                logger.info(f"  Stop Loss: ${signal.stop_loss:.2f}")
                logger.info(f"  Take Profit: ${signal.take_profit:.2f}")
                logger.info(f"  Entry Timeframe: {mtf_result.entry_timeframe}")
                logger.info(f"  Pattern: {signal.pattern_type}")
                logger.info(f"  Alignment: {mtf_result.overall_alignment:.2%}")
                logger.info(f"  Summary: {mtf_result.analysis_summary}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating MTF signal for {instrument}: {e}")
            return None
    
    def _convert_to_trading_signal(self, mtf_result: MTFAnalysisResult) -> TradingSignal:
        """Convert MTF analysis result to TradingSignal"""
        
        # Determine pattern type based on entry timeframe
        if mtf_result.entry_timeframe in ['1s', '5s', '10s', '15s', '30s']:
            pattern = 'scalp_micro'
        elif mtf_result.entry_timeframe in ['1m', '5m', '15m']:
            pattern = 'scalp'
        else:
            pattern = 'swing'
        
        # Build metadata
        metadata = {
            'entry_timeframe': mtf_result.entry_timeframe,
            'trend_alignment': mtf_result.trend_alignment,
            'momentum_alignment': mtf_result.momentum_alignment,
            'overall_alignment': mtf_result.overall_alignment,
            'micro_signals_count': len(mtf_result.micro_signals),
            'scalp_signals_count': len(mtf_result.scalp_signals),
            'swing_signals_count': len(mtf_result.swing_signals),
            'analysis_summary': mtf_result.analysis_summary,
            'mtf_enabled': True,
            'scalping_mode': self.scalping_mode,
        }
        
        # Add timeframe breakdown
        for tf_signal in mtf_result.micro_signals + mtf_result.scalp_signals + mtf_result.swing_signals:
            if tf_signal.direction:
                metadata[f'{tf_signal.timeframe}_direction'] = tf_signal.direction
                metadata[f'{tf_signal.timeframe}_confidence'] = tf_signal.confidence
        
        signal = TradingSignal(
            instrument=mtf_result.instrument,
            side=mtf_result.overall_direction,
            entry_price=mtf_result.entry_price,
            stop_loss=mtf_result.stop_loss,
            take_profit=mtf_result.take_profit,
            confidence=mtf_result.overall_confidence,
            timeframe=mtf_result.entry_timeframe,
            pattern_type=pattern,
            timestamp=mtf_result.timestamp,
            metadata=metadata
        )
        
        return signal
    
    def get_scalping_signal(self, instrument: str) -> Optional[TradingSignal]:
        """
        Generate scalping signal (optimized for speed)
        Uses only micro and scalp timeframes
        
        Args:
            instrument: Trading pair
            
        Returns:
            TradingSignal or None
        """
        # Temporarily override timeframes for scalping
        original_timeframes = self.timeframes
        self.timeframes = ['1s', '5s', '10s', '15s', '30s', '1m', '5m', '15m']
        
        try:
            signal = self.generate_signal(instrument)
            return signal
        finally:
            self.timeframes = original_timeframes


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with scalping mode
    config = {
        'scalping_mode': True,
        'min_confidence': 0.70,
        'min_alignment': 0.65,
        'require_trend_alignment': True
    }
    
    generator = MTFSignalGenerator(config)
    
    # Generate signal
    signal = generator.generate_signal('BTC-USDT')
    
    if signal:
        print(f"\n" + "=" * 80)
        print(f"✅ MTF TRADING SIGNAL GENERATED")
        print(f"=" * 80)
        print(f"Instrument: {signal.instrument}")
        print(f"Side: {signal.side.upper()}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Entry: ${signal.entry_price:.2f}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Pattern: {signal.pattern}")
        print(f"\nMetadata:")
        for key, value in signal.metadata.items():
            print(f"  {key}: {value}")
        print(f"=" * 80)
    else:
        print(f"\n❌ No MTF signal generated")

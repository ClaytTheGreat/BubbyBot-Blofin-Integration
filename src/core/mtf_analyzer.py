"""
Multi-Timeframe Analysis (MTF) Module
Analyzes multiple timeframes for scalping and higher timeframe confirmation
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from core.market_cipher_analyzer import MarketCipherAnalyzer, MarketCipherSignal
from core.lux_algo_analyzer import LuxAlgoAnalyzer, LuxAlgoSignal

logger = logging.getLogger(__name__)


class TimeframeCategory(Enum):
    """Timeframe categories"""
    MICRO = "micro"  # 1s-30s for ultra-precise scalping
    SCALP = "scalp"  # 1m-15m for scalping
    SWING = "swing"  # 1h-4h for trend confirmation
    POSITION = "position"  # 1D+ for long-term trend


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe"""
    timeframe: str
    category: TimeframeCategory
    mc_signal: Optional[MarketCipherSignal]
    lux_signal: Optional[LuxAlgoSignal]
    direction: Optional[str]  # 'buy', 'sell', or None
    confidence: float
    timestamp: datetime


@dataclass
class MTFAnalysisResult:
    """Multi-timeframe analysis result"""
    instrument: str
    overall_direction: str  # 'buy' or 'sell'
    overall_confidence: float  # 0-1
    entry_timeframe: str  # Best timeframe for entry
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Timeframe breakdown
    micro_signals: List[TimeframeSignal]  # 1s-30s
    scalp_signals: List[TimeframeSignal]  # 1m-15m
    swing_signals: List[TimeframeSignal]  # 1h-4h
    
    # Alignment scores
    trend_alignment: float  # 0-1 (higher TF alignment)
    momentum_alignment: float  # 0-1 (lower TF alignment)
    overall_alignment: float  # 0-1 (all TFs)
    
    # Metadata
    timestamp: datetime
    analysis_summary: str


class MultiTimeframeAnalyzer:
    """
    Multi-Timeframe Analyzer
    Implements top-down analysis with scalping focus
    """
    
    # Timeframe definitions
    MICRO_TIMEFRAMES = ['1s', '5s', '10s', '15s', '30s']
    SCALP_TIMEFRAMES = ['1m', '5m', '15m']
    SWING_TIMEFRAMES = ['1h', '4h']
    
    # Timeframe weights (must sum to 1.0)
    TIMEFRAME_WEIGHTS = {
        # Micro timeframes (10% total) - for precise entry
        '1s': 0.02,
        '5s': 0.02,
        '10s': 0.02,
        '15s': 0.02,
        '30s': 0.02,
        
        # Scalp timeframes (60% total) - primary trading timeframes
        '1m': 0.15,
        '5m': 0.20,
        '15m': 0.25,
        
        # Swing timeframes (30% total) - trend confirmation
        '1h': 0.15,
        '4h': 0.15,
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize Multi-Timeframe Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize analyzers
        self.mc_analyzer = MarketCipherAnalyzer(config)
        self.lux_analyzer = LuxAlgoAnalyzer(config)
        
        # MTF settings
        self.min_alignment = self.config.get('min_alignment', 0.65)  # 65% alignment required
        self.require_trend_alignment = self.config.get('require_trend_alignment', True)
        self.scalping_mode = self.config.get('scalping_mode', True)
        
        # Timeframe selection
        if self.scalping_mode:
            self.active_timeframes = (
                self.MICRO_TIMEFRAMES + 
                self.SCALP_TIMEFRAMES + 
                self.SWING_TIMEFRAMES
            )
        else:
            # Non-scalping: skip micro timeframes
            self.active_timeframes = self.SCALP_TIMEFRAMES + self.SWING_TIMEFRAMES
        
        logger.info(f"Multi-Timeframe Analyzer initialized (scalping_mode={self.scalping_mode})")
        logger.info(f"Active timeframes: {', '.join(self.active_timeframes)}")
    
    def analyze(self, data_dict: Dict[str, pd.DataFrame], instrument: str) -> Optional[MTFAnalysisResult]:
        """
        Perform multi-timeframe analysis
        
        Args:
            data_dict: Dictionary of {timeframe: OHLCV DataFrame}
            instrument: Trading pair
            
        Returns:
            MTFAnalysisResult or None
        """
        try:
            logger.info(f"Starting MTF analysis for {instrument}")
            
            # Step 1: Analyze each timeframe
            timeframe_signals = self._analyze_all_timeframes(data_dict)
            
            if not timeframe_signals:
                logger.info("No signals generated across timeframes")
                return None
            
            # Step 2: Calculate alignment scores
            alignment_scores = self._calculate_alignment(timeframe_signals)
            
            # Step 3: Determine overall direction
            overall_direction = self._determine_direction(timeframe_signals, alignment_scores)
            
            if not overall_direction:
                logger.info("No clear direction from MTF analysis")
                return None
            
            # Step 4: Check alignment threshold
            if alignment_scores['overall'] < self.min_alignment:
                logger.info(f"Alignment too low: {alignment_scores['overall']:.2%} < {self.min_alignment:.2%}")
                return None
            
            # Step 5: Check trend alignment requirement
            if self.require_trend_alignment and alignment_scores['trend'] < 0.60:
                logger.info(f"Trend alignment too low: {alignment_scores['trend']:.2%}")
                return None
            
            # Step 6: Calculate overall confidence
            overall_confidence = self._calculate_confidence(timeframe_signals, alignment_scores)
            
            # Step 7: Determine best entry timeframe and levels
            entry_tf, entry_price, stop_loss, take_profit = self._calculate_entry_levels(
                timeframe_signals, overall_direction, data_dict
            )
            
            # Step 8: Categorize signals
            micro_signals = [s for s in timeframe_signals if s.category == TimeframeCategory.MICRO]
            scalp_signals = [s for s in timeframe_signals if s.category == TimeframeCategory.SCALP]
            swing_signals = [s for s in timeframe_signals if s.category == TimeframeCategory.SWING]
            
            # Step 9: Generate analysis summary
            summary = self._generate_summary(
                overall_direction, overall_confidence, alignment_scores,
                micro_signals, scalp_signals, swing_signals
            )
            
            result = MTFAnalysisResult(
                instrument=instrument,
                overall_direction=overall_direction,
                overall_confidence=overall_confidence,
                entry_timeframe=entry_tf,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                micro_signals=micro_signals,
                scalp_signals=scalp_signals,
                swing_signals=swing_signals,
                trend_alignment=alignment_scores['trend'],
                momentum_alignment=alignment_scores['momentum'],
                overall_alignment=alignment_scores['overall'],
                timestamp=datetime.now(),
                analysis_summary=summary
            )
            
            logger.info(f"✅ MTF Analysis complete: {overall_direction.upper()} "
                       f"(confidence: {overall_confidence:.2%}, alignment: {alignment_scores['overall']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in MTF analysis: {e}")
            return None
    
    def _analyze_all_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> List[TimeframeSignal]:
        """Analyze all available timeframes"""
        signals = []
        
        for timeframe in self.active_timeframes:
            if timeframe not in data_dict:
                logger.warning(f"Data not available for {timeframe}")
                continue
            
            data = data_dict[timeframe]
            
            if len(data) < 50:  # Minimum data requirement
                logger.warning(f"Insufficient data for {timeframe}: {len(data)} bars")
                continue
            
            try:
                # Analyze with both analyzers
                mc_signal = self.mc_analyzer.analyze(data, timeframe=timeframe)
                lux_signal = self.lux_analyzer.analyze(data, timeframe=timeframe)
                
                # Determine direction and confidence
                direction = None
                confidence = 0.0
                
                if mc_signal and lux_signal:
                    if mc_signal.direction == lux_signal.direction:
                        direction = mc_signal.direction
                        confidence = (mc_signal.confidence + lux_signal.confidence) / 2
                elif mc_signal:
                    direction = mc_signal.direction
                    confidence = mc_signal.confidence * 0.7  # Reduce confidence without confluence
                elif lux_signal:
                    direction = lux_signal.direction
                    confidence = lux_signal.confidence * 0.7
                
                # Categorize timeframe
                if timeframe in self.MICRO_TIMEFRAMES:
                    category = TimeframeCategory.MICRO
                elif timeframe in self.SCALP_TIMEFRAMES:
                    category = TimeframeCategory.SCALP
                elif timeframe in self.SWING_TIMEFRAMES:
                    category = TimeframeCategory.SWING
                else:
                    category = TimeframeCategory.POSITION
                
                signal = TimeframeSignal(
                    timeframe=timeframe,
                    category=category,
                    mc_signal=mc_signal,
                    lux_signal=lux_signal,
                    direction=direction,
                    confidence=confidence,
                    timestamp=datetime.now()
                )
                
                signals.append(signal)
                
                if direction:
                    logger.debug(f"  {timeframe}: {direction.upper()} ({confidence:.2%})")
                
            except Exception as e:
                logger.error(f"Error analyzing {timeframe}: {e}")
        
        return signals
    
    def _calculate_alignment(self, signals: List[TimeframeSignal]) -> Dict[str, float]:
        """Calculate alignment scores across timeframes"""
        
        # Separate by category
        micro_signals = [s for s in signals if s.category == TimeframeCategory.MICRO and s.direction]
        scalp_signals = [s for s in signals if s.category == TimeframeCategory.SCALP and s.direction]
        swing_signals = [s for s in signals if s.category == TimeframeCategory.SWING and s.direction]
        
        # Calculate trend alignment (higher timeframes)
        trend_alignment = 0.0
        if swing_signals:
            # Check if swing timeframes agree
            swing_directions = [s.direction for s in swing_signals]
            if swing_directions:
                most_common = max(set(swing_directions), key=swing_directions.count)
                trend_alignment = swing_directions.count(most_common) / len(swing_directions)
        
        # Calculate momentum alignment (lower timeframes)
        momentum_alignment = 0.0
        lower_tf_signals = scalp_signals + micro_signals
        if lower_tf_signals:
            lower_directions = [s.direction for s in lower_tf_signals]
            if lower_directions:
                most_common = max(set(lower_directions), key=lower_directions.count)
                momentum_alignment = lower_directions.count(most_common) / len(lower_directions)
        
        # Calculate overall alignment (all timeframes)
        overall_alignment = 0.0
        all_directions = [s.direction for s in signals if s.direction]
        if all_directions:
            most_common = max(set(all_directions), key=all_directions.count)
            overall_alignment = all_directions.count(most_common) / len(all_directions)
        
        return {
            'trend': trend_alignment,
            'momentum': momentum_alignment,
            'overall': overall_alignment
        }
    
    def _determine_direction(
        self, 
        signals: List[TimeframeSignal], 
        alignment: Dict[str, float]
    ) -> Optional[str]:
        """Determine overall trading direction"""
        
        # Count votes by direction
        buy_votes = sum(1 for s in signals if s.direction == 'buy')
        sell_votes = sum(1 for s in signals if s.direction == 'sell')
        
        if buy_votes == 0 and sell_votes == 0:
            return None
        
        # Determine direction
        if buy_votes > sell_votes:
            return 'buy'
        elif sell_votes > buy_votes:
            return 'sell'
        else:
            # Tie: use higher timeframe as tiebreaker
            swing_signals = [s for s in signals if s.category == TimeframeCategory.SWING and s.direction]
            if swing_signals:
                swing_directions = [s.direction for s in swing_signals]
                most_common = max(set(swing_directions), key=swing_directions.count)
                return most_common
            return None
    
    def _calculate_confidence(
        self, 
        signals: List[TimeframeSignal], 
        alignment: Dict[str, float]
    ) -> float:
        """Calculate overall confidence score"""
        
        # Weighted average of timeframe confidences
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for signal in signals:
            if signal.direction and signal.timeframe in self.TIMEFRAME_WEIGHTS:
                weight = self.TIMEFRAME_WEIGHTS[signal.timeframe]
                weighted_confidence += signal.confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = weighted_confidence / total_weight
        
        # Apply alignment bonus
        alignment_bonus = alignment['overall'] * 0.2  # Up to 20% bonus
        
        final_confidence = min(1.0, base_confidence + alignment_bonus)
        
        return final_confidence
    
    def _calculate_entry_levels(
        self,
        signals: List[TimeframeSignal],
        direction: str,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[str, float, float, float]:
        """Calculate entry price, stop loss, and take profit"""
        
        # Use lowest timeframe with signal for entry
        entry_timeframe = '15m'  # Default
        for tf in ['1s', '5s', '10s', '15s', '30s', '1m', '5m', '15m']:
            if tf in data_dict:
                entry_timeframe = tf
                break
        
        # Get current price from entry timeframe
        entry_data = data_dict[entry_timeframe]
        entry_price = entry_data['close'].iloc[-1]
        
        # Calculate stop loss and take profit
        # Use tighter stops for scalping
        if entry_timeframe in self.MICRO_TIMEFRAMES:
            # Ultra-tight stops for micro timeframes
            sl_pct = 0.001  # 0.1%
            tp_pct = 0.003  # 0.3% (3:1 R:R)
        elif entry_timeframe in self.SCALP_TIMEFRAMES:
            # Tight stops for scalping
            sl_pct = 0.005  # 0.5%
            tp_pct = 0.015  # 1.5% (3:1 R:R)
        else:
            # Normal stops for swing
            sl_pct = 0.02  # 2%
            tp_pct = 0.06  # 6% (3:1 R:R)
        
        if direction == 'buy':
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)
        
        return entry_timeframe, entry_price, stop_loss, take_profit
    
    def _generate_summary(
        self,
        direction: str,
        confidence: float,
        alignment: Dict[str, float],
        micro_signals: List[TimeframeSignal],
        scalp_signals: List[TimeframeSignal],
        swing_signals: List[TimeframeSignal]
    ) -> str:
        """Generate human-readable analysis summary"""
        
        parts = []
        
        # Overall
        parts.append(f"MTF {direction.upper()} signal ({confidence:.1%} confidence)")
        
        # Alignment
        parts.append(f"Alignment: {alignment['overall']:.1%} overall, "
                    f"{alignment['trend']:.1%} trend, {alignment['momentum']:.1%} momentum")
        
        # Swing timeframes (trend)
        if swing_signals:
            swing_dirs = [s.direction for s in swing_signals if s.direction]
            buy_count = swing_dirs.count('buy')
            sell_count = swing_dirs.count('sell')
            parts.append(f"Swing TFs: {buy_count} buy, {sell_count} sell")
        
        # Scalp timeframes (setup)
        if scalp_signals:
            scalp_dirs = [s.direction for s in scalp_signals if s.direction]
            buy_count = scalp_dirs.count('buy')
            sell_count = scalp_dirs.count('sell')
            parts.append(f"Scalp TFs: {buy_count} buy, {sell_count} sell")
        
        # Micro timeframes (entry)
        if micro_signals:
            micro_dirs = [s.direction for s in micro_signals if s.direction]
            buy_count = micro_dirs.count('buy')
            sell_count = micro_dirs.count('sell')
            parts.append(f"Micro TFs: {buy_count} buy, {sell_count} sell")
        
        return " | ".join(parts)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download data for multiple timeframes
    ticker = yf.Ticker("BTC-USD")
    
    data_dict = {
        '1m': ticker.history(period="1d", interval="1m"),
        '5m': ticker.history(period="5d", interval="5m"),
        '15m': ticker.history(period="1mo", interval="15m"),
        '1h': ticker.history(period="3mo", interval="1h"),
        '4h': ticker.history(period="1y", interval="1d")  # Approximate 4h with daily
    }
    
    # Standardize column names
    for tf in data_dict:
        data_dict[tf].columns = data_dict[tf].columns.str.lower()
    
    # Analyze
    config = {'scalping_mode': True, 'min_alignment': 0.65}
    analyzer = MultiTimeframeAnalyzer(config)
    result = analyzer.analyze(data_dict, 'BTC-USDT')
    
    if result:
        print(f"\n✅ MTF Analysis Result:")
        print(f"  Direction: {result.overall_direction.upper()}")
        print(f"  Confidence: {result.overall_confidence:.2%}")
        print(f"  Entry TF: {result.entry_timeframe}")
        print(f"  Entry: ${result.entry_price:.2f}")
        print(f"  Stop Loss: ${result.stop_loss:.2f}")
        print(f"  Take Profit: ${result.take_profit:.2f}")
        print(f"  Alignment: {result.overall_alignment:.2%}")
        print(f"  Summary: {result.analysis_summary}")
    else:
        print("\n❌ No MTF signal generated")

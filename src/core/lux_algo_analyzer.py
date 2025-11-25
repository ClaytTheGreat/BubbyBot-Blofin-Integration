"""
Lux Algo Analyzer
Analyzes Lux Algo indicators for Smart Money Concepts (SMC), order blocks,
premium/discount zones, and support/resistance levels
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LuxSignalType(Enum):
    """Lux Algo signal types"""
    ORDER_BLOCK_BULLISH = "order_block_bullish"
    ORDER_BLOCK_BEARISH = "order_block_bearish"
    MARKET_STRUCTURE_BREAK = "market_structure_break"
    PREMIUM_ZONE = "premium_zone"
    DISCOUNT_ZONE = "discount_zone"
    SUPPORT_LEVEL = "support_level"
    RESISTANCE_LEVEL = "resistance_level"
    LIQUIDITY_GRAB = "liquidity_grab"
    FAIR_VALUE_GAP = "fair_value_gap"


@dataclass
class OrderBlock:
    """Order block data structure"""
    type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    volume: float
    timestamp: datetime
    strength: float  # 0-1


@dataclass
class LuxAlgoSignal:
    """Lux Algo analysis result"""
    signal_type: LuxSignalType
    direction: str  # 'buy' or 'sell'
    confidence: float  # 0-1
    timeframe: str
    price: float
    entry_zone: Tuple[float, float]  # (low, high)
    stop_loss: float
    take_profit: float
    indicators: Dict[str, float]
    timestamp: datetime
    description: str


class LuxAlgoAnalyzer:
    """
    Lux Algo Analyzer
    Implements Smart Money Concepts (SMC) and price action analysis
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Lux Algo Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Lux Algo parameters
        self.lux_params = {
            'order_block_lookback': 20,
            'structure_lookback': 50,
            'volume_threshold': 1.5,  # 1.5x average volume
            'premium_discount_threshold': 0.5,  # 50% of range
            'fair_value_gap_min': 0.001  # 0.1% minimum gap
        }
        
        # Track order blocks
        self.bullish_order_blocks: List[OrderBlock] = []
        self.bearish_order_blocks: List[OrderBlock] = []
        
        logger.info("Lux Algo Analyzer initialized")
    
    def analyze(self, data: pd.DataFrame, timeframe: str = "15m") -> Optional[LuxAlgoSignal]:
        """
        Analyze market data using Lux Algo concepts
        
        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume
            timeframe: Timeframe being analyzed
            
        Returns:
            LuxAlgoSignal or None
        """
        try:
            if len(data) < 100:
                logger.warning(f"Insufficient data for analysis: {len(data)} bars")
                return None
            
            # Update order blocks
            self._identify_order_blocks(data)
            
            # Detect signals
            signals = []
            
            # Check for order block interactions
            ob_signal = self._detect_order_block_signal(data)
            if ob_signal:
                signals.append(ob_signal)
            
            # Check for market structure breaks
            msb_signal = self._detect_market_structure_break(data)
            if msb_signal:
                signals.append(msb_signal)
            
            # Check for premium/discount zones
            zone_signal = self._detect_premium_discount_zones(data)
            if zone_signal:
                signals.append(zone_signal)
            
            # Check for support/resistance levels
            sr_signal = self._detect_support_resistance(data)
            if sr_signal:
                signals.append(sr_signal)
            
            # Check for fair value gaps
            fvg_signal = self._detect_fair_value_gap(data)
            if fvg_signal:
                signals.append(fvg_signal)
            
            # Check for liquidity grabs
            liq_signal = self._detect_liquidity_grab(data)
            if liq_signal:
                signals.append(liq_signal)
            
            # Return highest confidence signal
            if signals:
                best_signal = max(signals, key=lambda s: s.confidence)
                if best_signal.confidence >= self.min_confidence:
                    logger.info(f"Lux Algo Signal: {best_signal.signal_type.value} "
                              f"({best_signal.direction}) - Confidence: {best_signal.confidence:.2%}")
                    return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Lux Algo analysis: {e}")
            return None
    
    def _identify_order_blocks(self, data: pd.DataFrame):
        """Identify bullish and bearish order blocks"""
        try:
            lookback = self.lux_params['order_block_lookback']
            avg_volume = data['volume'].rolling(window=20).mean()
            
            # Clear old order blocks
            self.bullish_order_blocks.clear()
            self.bearish_order_blocks.clear()
            
            for i in range(lookback, len(data)):
                current_volume = data['volume'].iloc[i]
                current_close = data['close'].iloc[i]
                current_open = data['open'].iloc[i]
                
                # High volume candle
                if current_volume > avg_volume.iloc[i] * self.lux_params['volume_threshold']:
                    
                    # Bullish order block: Strong buying, then price moves up
                    if current_close > current_open:
                        # Check if price moved up after this candle
                        if i < len(data) - 5:
                            future_high = data['high'].iloc[i+1:i+6].max()
                            if future_high > current_close * 1.01:  # 1% move up
                                strength = min(1.0, current_volume / (avg_volume.iloc[i] * 2))
                                self.bullish_order_blocks.append(OrderBlock(
                                    type='bullish',
                                    high=data['high'].iloc[i],
                                    low=data['low'].iloc[i],
                                    volume=current_volume,
                                    timestamp=data.index[i],
                                    strength=strength
                                ))
                    
                    # Bearish order block: Strong selling, then price moves down
                    elif current_close < current_open:
                        if i < len(data) - 5:
                            future_low = data['low'].iloc[i+1:i+6].min()
                            if future_low < current_close * 0.99:  # 1% move down
                                strength = min(1.0, current_volume / (avg_volume.iloc[i] * 2))
                                self.bearish_order_blocks.append(OrderBlock(
                                    type='bearish',
                                    high=data['high'].iloc[i],
                                    low=data['low'].iloc[i],
                                    volume=current_volume,
                                    timestamp=data.index[i],
                                    strength=strength
                                ))
            
            logger.debug(f"Identified {len(self.bullish_order_blocks)} bullish and "
                        f"{len(self.bearish_order_blocks)} bearish order blocks")
            
        except Exception as e:
            logger.error(f"Error identifying order blocks: {e}")
    
    def _detect_order_block_signal(self, data: pd.DataFrame) -> Optional[LuxAlgoSignal]:
        """Detect price interaction with order blocks"""
        current_price = data['close'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_high = data['high'].iloc[-1]
        
        # Check bullish order blocks (support)
        for ob in self.bullish_order_blocks[-5:]:  # Check last 5 blocks
            if ob.low <= current_low <= ob.high:
                # Price is testing bullish order block
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.ORDER_BLOCK_BULLISH,
                    direction='buy',
                    confidence=0.75 + (ob.strength * 0.15),
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(ob.low, ob.high),
                    stop_loss=ob.low * 0.998,  # 0.2% below
                    take_profit=current_price * 1.03,  # 3% above
                    indicators={'volume': ob.volume, 'strength': ob.strength},
                    timestamp=datetime.now(),
                    description=f"Bullish order block support at {ob.low:.2f}-{ob.high:.2f}"
                )
        
        # Check bearish order blocks (resistance)
        for ob in self.bearish_order_blocks[-5:]:
            if ob.low <= current_high <= ob.high:
                # Price is testing bearish order block
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.ORDER_BLOCK_BEARISH,
                    direction='sell',
                    confidence=0.75 + (ob.strength * 0.15),
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(ob.low, ob.high),
                    stop_loss=ob.high * 1.002,  # 0.2% above
                    take_profit=current_price * 0.97,  # 3% below
                    indicators={'volume': ob.volume, 'strength': ob.strength},
                    timestamp=datetime.now(),
                    description=f"Bearish order block resistance at {ob.low:.2f}-{ob.high:.2f}"
                )
        
        return None
    
    def _detect_market_structure_break(self, data: pd.DataFrame) -> Optional[LuxAlgoSignal]:
        """Detect market structure breaks (BOS)"""
        try:
            lookback = self.lux_params['structure_lookback']
            recent_data = data.tail(lookback)
            
            # Find swing highs and lows
            swing_period = 10
            swing_highs = recent_data['high'].rolling(window=swing_period, center=True).max()
            swing_lows = recent_data['low'].rolling(window=swing_period, center=True).min()
            
            current_price = data['close'].iloc[-1]
            
            # Bullish BOS: Price breaks above recent swing high
            recent_high = swing_highs.iloc[-20:].max()
            if current_price > recent_high * 1.005:  # 0.5% above
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.MARKET_STRUCTURE_BREAK,
                    direction='buy',
                    confidence=0.82,
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(recent_high, current_price),
                    stop_loss=recent_high * 0.995,
                    take_profit=current_price * 1.04,
                    indicators={'structure_level': recent_high},
                    timestamp=datetime.now(),
                    description=f"Bullish market structure break above {recent_high:.2f}"
                )
            
            # Bearish BOS: Price breaks below recent swing low
            recent_low = swing_lows.iloc[-20:].min()
            if current_price < recent_low * 0.995:  # 0.5% below
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.MARKET_STRUCTURE_BREAK,
                    direction='sell',
                    confidence=0.82,
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(current_price, recent_low),
                    stop_loss=recent_low * 1.005,
                    take_profit=current_price * 0.96,
                    indicators={'structure_level': recent_low},
                    timestamp=datetime.now(),
                    description=f"Bearish market structure break below {recent_low:.2f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting market structure break: {e}")
            return None
    
    def _detect_premium_discount_zones(self, data: pd.DataFrame) -> Optional[LuxAlgoSignal]:
        """Detect premium and discount zones"""
        try:
            # Calculate range
            lookback = 50
            recent_high = data['high'].tail(lookback).max()
            recent_low = data['low'].tail(lookback).min()
            range_size = recent_high - recent_low
            
            # Calculate equilibrium (50% level)
            equilibrium = recent_low + (range_size * 0.5)
            
            # Premium zone: 50-100% of range
            premium_low = equilibrium
            premium_high = recent_high
            
            # Discount zone: 0-50% of range
            discount_low = recent_low
            discount_high = equilibrium
            
            current_price = data['close'].iloc[-1]
            
            # In discount zone (buy opportunity)
            if discount_low <= current_price <= discount_high:
                position_in_discount = (current_price - discount_low) / (discount_high - discount_low)
                confidence = 0.70 + (0.15 * (1 - position_in_discount))  # Higher confidence near bottom
                
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.DISCOUNT_ZONE,
                    direction='buy',
                    confidence=confidence,
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(discount_low, discount_high),
                    stop_loss=discount_low * 0.995,
                    take_profit=equilibrium,
                    indicators={
                        'equilibrium': equilibrium,
                        'discount_low': discount_low,
                        'position': position_in_discount
                    },
                    timestamp=datetime.now(),
                    description=f"Price in discount zone ({position_in_discount:.1%} from bottom)"
                )
            
            # In premium zone (sell opportunity)
            if premium_low <= current_price <= premium_high:
                position_in_premium = (current_price - premium_low) / (premium_high - premium_low)
                confidence = 0.70 + (0.15 * position_in_premium)  # Higher confidence near top
                
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.PREMIUM_ZONE,
                    direction='sell',
                    confidence=confidence,
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(premium_low, premium_high),
                    stop_loss=premium_high * 1.005,
                    take_profit=equilibrium,
                    indicators={
                        'equilibrium': equilibrium,
                        'premium_high': premium_high,
                        'position': position_in_premium
                    },
                    timestamp=datetime.now(),
                    description=f"Price in premium zone ({position_in_premium:.1%} from bottom)"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting premium/discount zones: {e}")
            return None
    
    def _detect_support_resistance(self, data: pd.DataFrame) -> Optional[LuxAlgoSignal]:
        """Detect support and resistance levels"""
        try:
            # Find recent pivot points
            window = 10
            highs = data['high'].rolling(window=window, center=True).max()
            lows = data['low'].rolling(window=window, center=True).min()
            
            # Identify levels where price has bounced multiple times
            current_price = data['close'].iloc[-1]
            tolerance = current_price * 0.005  # 0.5% tolerance
            
            # Check for support (price near recent lows)
            recent_lows = lows.tail(50).dropna().unique()
            for level in recent_lows:
                if abs(current_price - level) < tolerance:
                    # Count touches
                    touches = sum(abs(data['low'].tail(50) - level) < tolerance)
                    if touches >= 2:
                        return LuxAlgoSignal(
                            signal_type=LuxSignalType.SUPPORT_LEVEL,
                            direction='buy',
                            confidence=0.72 + (min(touches, 5) * 0.03),
                            timeframe="15m",
                            price=current_price,
                            entry_zone=(level * 0.998, level * 1.002),
                            stop_loss=level * 0.995,
                            take_profit=current_price * 1.03,
                            indicators={'level': level, 'touches': touches},
                            timestamp=datetime.now(),
                            description=f"Support level at {level:.2f} ({touches} touches)"
                        )
            
            # Check for resistance (price near recent highs)
            recent_highs = highs.tail(50).dropna().unique()
            for level in recent_highs:
                if abs(current_price - level) < tolerance:
                    touches = sum(abs(data['high'].tail(50) - level) < tolerance)
                    if touches >= 2:
                        return LuxAlgoSignal(
                            signal_type=LuxSignalType.RESISTANCE_LEVEL,
                            direction='sell',
                            confidence=0.72 + (min(touches, 5) * 0.03),
                            timeframe="15m",
                            price=current_price,
                            entry_zone=(level * 0.998, level * 1.002),
                            stop_loss=level * 1.005,
                            take_profit=current_price * 0.97,
                            indicators={'level': level, 'touches': touches},
                            timestamp=datetime.now(),
                            description=f"Resistance level at {level:.2f} ({touches} touches)"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return None
    
    def _detect_fair_value_gap(self, data: pd.DataFrame) -> Optional[LuxAlgoSignal]:
        """Detect fair value gaps (FVG)"""
        try:
            if len(data) < 3:
                return None
            
            # Check last 3 candles for gap
            candle_1 = data.iloc[-3]
            candle_2 = data.iloc[-2]
            candle_3 = data.iloc[-1]
            
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if candle_1['high'] < candle_3['low']:
                gap_size = (candle_3['low'] - candle_1['high']) / candle_1['high']
                if gap_size >= self.lux_params['fair_value_gap_min']:
                    return LuxAlgoSignal(
                        signal_type=LuxSignalType.FAIR_VALUE_GAP,
                        direction='buy',
                        confidence=0.78,
                        timeframe="15m",
                        price=data['close'].iloc[-1],
                        entry_zone=(candle_1['high'], candle_3['low']),
                        stop_loss=candle_1['high'] * 0.995,
                        take_profit=data['close'].iloc[-1] * 1.03,
                        indicators={'gap_size': gap_size},
                        timestamp=datetime.now(),
                        description=f"Bullish fair value gap ({gap_size:.2%})"
                    )
            
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            if candle_1['low'] > candle_3['high']:
                gap_size = (candle_1['low'] - candle_3['high']) / candle_1['low']
                if gap_size >= self.lux_params['fair_value_gap_min']:
                    return LuxAlgoSignal(
                        signal_type=LuxSignalType.FAIR_VALUE_GAP,
                        direction='sell',
                        confidence=0.78,
                        timeframe="15m",
                        price=data['close'].iloc[-1],
                        entry_zone=(candle_3['high'], candle_1['low']),
                        stop_loss=candle_1['low'] * 1.005,
                        take_profit=data['close'].iloc[-1] * 0.97,
                        indicators={'gap_size': gap_size},
                        timestamp=datetime.now(),
                        description=f"Bearish fair value gap ({gap_size:.2%})"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting fair value gap: {e}")
            return None
    
    def _detect_liquidity_grab(self, data: pd.DataFrame) -> Optional[LuxAlgoSignal]:
        """Detect liquidity grabs (stop hunts)"""
        try:
            if len(data) < 20:
                return None
            
            # Look for recent swing high/low that was briefly broken
            recent_high = data['high'].tail(20).max()
            recent_low = data['low'].tail(20).min()
            
            current_price = data['close'].iloc[-1]
            prev_high = data['high'].iloc[-2]
            prev_low = data['low'].iloc[-2]
            
            # Bullish liquidity grab: Wick below recent low, then reversal
            if prev_low < recent_low * 0.998 and current_price > prev_low * 1.005:
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.LIQUIDITY_GRAB,
                    direction='buy',
                    confidence=0.80,
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(prev_low, current_price),
                    stop_loss=prev_low * 0.995,
                    take_profit=current_price * 1.04,
                    indicators={'grabbed_level': prev_low},
                    timestamp=datetime.now(),
                    description=f"Bullish liquidity grab at {prev_low:.2f}"
                )
            
            # Bearish liquidity grab: Wick above recent high, then reversal
            if prev_high > recent_high * 1.002 and current_price < prev_high * 0.995:
                return LuxAlgoSignal(
                    signal_type=LuxSignalType.LIQUIDITY_GRAB,
                    direction='sell',
                    confidence=0.80,
                    timeframe="15m",
                    price=current_price,
                    entry_zone=(current_price, prev_high),
                    stop_loss=prev_high * 1.005,
                    take_profit=current_price * 0.96,
                    indicators={'grabbed_level': prev_high},
                    timestamp=datetime.now(),
                    description=f"Bearish liquidity grab at {prev_high:.2f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting liquidity grab: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period="1mo", interval="15m")
    data.columns = data.columns.str.lower()
    
    # Analyze
    analyzer = LuxAlgoAnalyzer()
    signal = analyzer.analyze(data)
    
    if signal:
        print(f"Signal: {signal.signal_type.value}")
        print(f"Direction: {signal.direction}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Entry Zone: {signal.entry_zone}")
        print(f"Stop Loss: {signal.stop_loss:.2f}")
        print(f"Take Profit: {signal.take_profit:.2f}")
        print(f"Description: {signal.description}")
    else:
        print("No signal detected")

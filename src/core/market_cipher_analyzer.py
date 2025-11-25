"""
Market Cipher Analyzer
Analyzes Market Cipher indicators (A, B, SR, DBSI) for signal generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MCSignalType(Enum):
    """Market Cipher signal types"""
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    BLOOD_DIAMOND = "blood_diamond"
    YELLOW_X = "yellow_x"
    MONEY_FLOW_REVERSAL = "money_flow_reversal"
    WAVE_TREND_CROSS = "wave_trend_cross"
    SQUEEZE_RELEASE = "squeeze_release"
    TREND_ALIGNMENT = "trend_alignment"


@dataclass
class MarketCipherSignal:
    """Market Cipher analysis result"""
    signal_type: MCSignalType
    direction: str  # 'buy' or 'sell'
    confidence: float  # 0-1
    timeframe: str
    price: float
    indicators: Dict[str, float]
    timestamp: datetime
    description: str


class MarketCipherAnalyzer:
    """
    Market Cipher Analyzer
    Analyzes MC-A, MC-B, MC-SR, and MC-DBSI indicators
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Market Cipher Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Market Cipher parameters
        self.mc_params = {
            'rsi_period': 14,
            'wave_trend_n1': 10,
            'wave_trend_n2': 21,
            'money_flow_length': 14,
            'divergence_lookback': 50,
            'ema_periods': [9, 21, 55, 100, 200]
        }
        
        logger.info("Market Cipher Analyzer initialized")
    
    def analyze(self, data: pd.DataFrame, timeframe: str = "15m") -> Optional[MarketCipherSignal]:
        """
        Analyze market data using Market Cipher indicators
        
        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume
            timeframe: Timeframe being analyzed
            
        Returns:
            MarketCipherSignal or None
        """
        try:
            if len(data) < 200:
                logger.warning(f"Insufficient data for analysis: {len(data)} bars")
                return None
            
            # Calculate all Market Cipher indicators
            mc_a = self._calculate_mc_a(data)
            mc_b = self._calculate_mc_b(data)
            mc_sr = self._calculate_mc_sr(data)
            
            # Detect signals
            signals = []
            
            # Check for divergences
            divergence_signal = self._detect_divergence(data, mc_b)
            if divergence_signal:
                signals.append(divergence_signal)
            
            # Check for blood diamond (strong buy signal)
            blood_diamond = self._detect_blood_diamond(mc_a, mc_b)
            if blood_diamond:
                signals.append(blood_diamond)
            
            # Check for wave trend crosses
            wave_cross = self._detect_wave_trend_cross(mc_b)
            if wave_cross:
                signals.append(wave_cross)
            
            # Check for money flow reversals
            money_flow_signal = self._detect_money_flow_reversal(mc_b)
            if money_flow_signal:
                signals.append(money_flow_signal)
            
            # Check for trend alignment
            trend_signal = self._detect_trend_alignment(mc_a, mc_b, mc_sr)
            if trend_signal:
                signals.append(trend_signal)
            
            # Return highest confidence signal
            if signals:
                best_signal = max(signals, key=lambda s: s.confidence)
                if best_signal.confidence >= self.min_confidence:
                    logger.info(f"MC Signal: {best_signal.signal_type.value} "
                              f"({best_signal.direction}) - Confidence: {best_signal.confidence:.2%}")
                    return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Market Cipher analysis: {e}")
            return None
    
    def _calculate_mc_a(self, data: pd.DataFrame) -> Dict:
        """Calculate Market Cipher A indicators (Trend)"""
        mc_a = {}
        
        # Calculate EMAs
        for period in self.mc_params['ema_periods']:
            mc_a[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean().iloc[-1]
        
        # EMA alignment (trend strength)
        emas = [mc_a[f'ema_{p}'] for p in self.mc_params['ema_periods']]
        mc_a['ema_aligned_bullish'] = all(emas[i] > emas[i+1] for i in range(len(emas)-1))
        mc_a['ema_aligned_bearish'] = all(emas[i] < emas[i+1] for i in range(len(emas)-1))
        
        # Trend strength (0-1)
        current_price = data['close'].iloc[-1]
        if mc_a['ema_aligned_bullish']:
            mc_a['trend_strength'] = min(1.0, (current_price - emas[-1]) / emas[-1] * 10)
        elif mc_a['ema_aligned_bearish']:
            mc_a['trend_strength'] = min(1.0, (emas[-1] - current_price) / emas[-1] * 10)
        else:
            mc_a['trend_strength'] = 0.5
        
        return mc_a
    
    def _calculate_mc_b(self, data: pd.DataFrame) -> Dict:
        """Calculate Market Cipher B indicators (Momentum & Money Flow)"""
        mc_b = {}
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.mc_params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.mc_params['rsi_period']).mean()
        rs = gain / loss
        mc_b['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Wave Trend (simplified)
        hlc3 = (data['high'] + data['low'] + data['close']) / 3
        esa = hlc3.ewm(span=self.mc_params['wave_trend_n1'], adjust=False).mean()
        d = (hlc3 - esa).abs().ewm(span=self.mc_params['wave_trend_n1'], adjust=False).mean()
        ci = (hlc3 - esa) / (0.015 * d)
        wt1 = ci.ewm(span=self.mc_params['wave_trend_n2'], adjust=False).mean()
        wt2 = wt1.rolling(window=4).mean()
        
        mc_b['wt1'] = wt1.iloc[-1]
        mc_b['wt2'] = wt2.iloc[-1]
        mc_b['wt_cross_bullish'] = wt1.iloc[-2] < wt2.iloc[-2] and wt1.iloc[-1] > wt2.iloc[-1]
        mc_b['wt_cross_bearish'] = wt1.iloc[-2] > wt2.iloc[-2] and wt1.iloc[-1] < wt2.iloc[-1]
        
        # Money Flow (MFI-like)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=self.mc_params['money_flow_length']).sum()
        negative_mf = negative_flow.rolling(window=self.mc_params['money_flow_length']).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        mc_b['mfi'] = mfi.iloc[-1]
        
        # Money flow strength
        mc_b['money_flow_strength'] = abs(mc_b['mfi'] - 50) / 50
        
        return mc_b
    
    def _calculate_mc_sr(self, data: pd.DataFrame) -> Dict:
        """Calculate Market Cipher SR indicators (Support/Resistance)"""
        mc_sr = {}
        
        # Identify recent highs and lows
        window = 20
        mc_sr['resistance'] = data['high'].rolling(window=window).max().iloc[-1]
        mc_sr['support'] = data['low'].rolling(window=window).min().iloc[-1]
        
        current_price = data['close'].iloc[-1]
        price_range = mc_sr['resistance'] - mc_sr['support']
        
        # Position in range (0 = support, 1 = resistance)
        mc_sr['position_in_range'] = (current_price - mc_sr['support']) / price_range if price_range > 0 else 0.5
        
        # Near support/resistance
        mc_sr['near_support'] = mc_sr['position_in_range'] < 0.1
        mc_sr['near_resistance'] = mc_sr['position_in_range'] > 0.9
        
        return mc_sr
    
    def _detect_divergence(self, data: pd.DataFrame, mc_b: Dict) -> Optional[MarketCipherSignal]:
        """Detect bullish/bearish divergences"""
        try:
            lookback = min(self.mc_params['divergence_lookback'], len(data))
            recent_data = data.tail(lookback)
            
            # Find price lows and highs
            price_lows = recent_data['low'].rolling(window=5, center=True).min()
            price_highs = recent_data['high'].rolling(window=5, center=True).max()
            
            # Bullish divergence: price making lower lows, but RSI making higher lows
            if mc_b['rsi'] < 40:  # Oversold region
                price_trend = recent_data['low'].iloc[-1] < recent_data['low'].iloc[-20]
                rsi_series = recent_data['close'].rolling(window=14).apply(
                    lambda x: 100 - (100 / (1 + (x[x > x.shift(1)].mean() / abs(x[x < x.shift(1)].mean()))))
                )
                rsi_trend = rsi_series.iloc[-1] > rsi_series.iloc[-20]
                
                if price_trend and rsi_trend:
                    return MarketCipherSignal(
                        signal_type=MCSignalType.BULLISH_DIVERGENCE,
                        direction='buy',
                        confidence=0.85,
                        timeframe="15m",
                        price=data['close'].iloc[-1],
                        indicators={'rsi': mc_b['rsi'], 'wt1': mc_b['wt1']},
                        timestamp=datetime.now(),
                        description="Bullish divergence detected: Price lower low, RSI higher low"
                    )
            
            # Bearish divergence: price making higher highs, but RSI making lower highs
            if mc_b['rsi'] > 60:  # Overbought region
                price_trend = recent_data['high'].iloc[-1] > recent_data['high'].iloc[-20]
                rsi_series = recent_data['close'].rolling(window=14).apply(
                    lambda x: 100 - (100 / (1 + (x[x > x.shift(1)].mean() / abs(x[x < x.shift(1)].mean()))))
                )
                rsi_trend = rsi_series.iloc[-1] < rsi_series.iloc[-20]
                
                if price_trend and rsi_trend:
                    return MarketCipherSignal(
                        signal_type=MCSignalType.BEARISH_DIVERGENCE,
                        direction='sell',
                        confidence=0.85,
                        timeframe="15m",
                        price=data['close'].iloc[-1],
                        indicators={'rsi': mc_b['rsi'], 'wt1': mc_b['wt1']},
                        timestamp=datetime.now(),
                        description="Bearish divergence detected: Price higher high, RSI lower high"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting divergence: {e}")
            return None
    
    def _detect_blood_diamond(self, mc_a: Dict, mc_b: Dict) -> Optional[MarketCipherSignal]:
        """Detect blood diamond signal (strong buy)"""
        # Blood diamond: Oversold RSI + Wave trend cross + EMA alignment
        if (mc_b['rsi'] < 30 and 
            mc_b['wt_cross_bullish'] and 
            mc_a['ema_aligned_bullish']):
            
            return MarketCipherSignal(
                signal_type=MCSignalType.BLOOD_DIAMOND,
                direction='buy',
                confidence=0.95,
                timeframe="15m",
                price=0.0,  # Will be filled by caller
                indicators={'rsi': mc_b['rsi'], 'wt1': mc_b['wt1'], 'trend': mc_a['trend_strength']},
                timestamp=datetime.now(),
                description="Blood Diamond: Strong buy signal with trend alignment"
            )
        
        return None
    
    def _detect_wave_trend_cross(self, mc_b: Dict) -> Optional[MarketCipherSignal]:
        """Detect wave trend crosses"""
        if mc_b['wt_cross_bullish'] and mc_b['wt1'] < -40:
            return MarketCipherSignal(
                signal_type=MCSignalType.WAVE_TREND_CROSS,
                direction='buy',
                confidence=0.75,
                timeframe="15m",
                price=0.0,
                indicators={'wt1': mc_b['wt1'], 'wt2': mc_b['wt2']},
                timestamp=datetime.now(),
                description="Bullish wave trend cross in oversold region"
            )
        
        if mc_b['wt_cross_bearish'] and mc_b['wt1'] > 40:
            return MarketCipherSignal(
                signal_type=MCSignalType.WAVE_TREND_CROSS,
                direction='sell',
                confidence=0.75,
                timeframe="15m",
                price=0.0,
                indicators={'wt1': mc_b['wt1'], 'wt2': mc_b['wt2']},
                timestamp=datetime.now(),
                description="Bearish wave trend cross in overbought region"
            )
        
        return None
    
    def _detect_money_flow_reversal(self, mc_b: Dict) -> Optional[MarketCipherSignal]:
        """Detect money flow reversals"""
        # Strong money flow in oversold/overbought regions
        if mc_b['mfi'] < 20 and mc_b['rsi'] < 35:
            return MarketCipherSignal(
                signal_type=MCSignalType.MONEY_FLOW_REVERSAL,
                direction='buy',
                confidence=0.80,
                timeframe="15m",
                price=0.0,
                indicators={'mfi': mc_b['mfi'], 'rsi': mc_b['rsi']},
                timestamp=datetime.now(),
                description="Money flow reversal: Oversold with weak money flow"
            )
        
        if mc_b['mfi'] > 80 and mc_b['rsi'] > 65:
            return MarketCipherSignal(
                signal_type=MCSignalType.MONEY_FLOW_REVERSAL,
                direction='sell',
                confidence=0.80,
                timeframe="15m",
                price=0.0,
                indicators={'mfi': mc_b['mfi'], 'rsi': mc_b['rsi']},
                timestamp=datetime.now(),
                description="Money flow reversal: Overbought with strong money flow"
            )
        
        return None
    
    def _detect_trend_alignment(self, mc_a: Dict, mc_b: Dict, mc_sr: Dict) -> Optional[MarketCipherSignal]:
        """Detect trend alignment signals"""
        # Bullish: EMA aligned + RSI recovering + near support
        if (mc_a['ema_aligned_bullish'] and 
            35 < mc_b['rsi'] < 55 and 
            mc_sr['near_support']):
            
            return MarketCipherSignal(
                signal_type=MCSignalType.TREND_ALIGNMENT,
                direction='buy',
                confidence=0.82,
                timeframe="15m",
                price=0.0,
                indicators={
                    'trend': mc_a['trend_strength'],
                    'rsi': mc_b['rsi'],
                    'position': mc_sr['position_in_range']
                },
                timestamp=datetime.now(),
                description="Trend alignment: Bullish trend with support bounce"
            )
        
        # Bearish: EMA aligned + RSI weakening + near resistance
        if (mc_a['ema_aligned_bearish'] and 
            45 < mc_b['rsi'] < 65 and 
            mc_sr['near_resistance']):
            
            return MarketCipherSignal(
                signal_type=MCSignalType.TREND_ALIGNMENT,
                direction='sell',
                confidence=0.82,
                timeframe="15m",
                price=0.0,
                indicators={
                    'trend': mc_a['trend_strength'],
                    'rsi': mc_b['rsi'],
                    'position': mc_sr['position_in_range']
                },
                timestamp=datetime.now(),
                description="Trend alignment: Bearish trend with resistance rejection"
            )
        
        return None


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period="1mo", interval="15m")
    data.columns = data.columns.str.lower()
    
    # Analyze
    analyzer = MarketCipherAnalyzer()
    signal = analyzer.analyze(data)
    
    if signal:
        print(f"Signal: {signal.signal_type.value}")
        print(f"Direction: {signal.direction}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Description: {signal.description}")
    else:
        print("No signal detected")

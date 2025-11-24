#!/usr/bin/env python3
"""
BubbyBot Enhanced V2 - Comprehensive Market Cipher & Lux Algo Trading System
Integrated with Pattern Book, Success Rate Analytics, and Optimized Settings
"""

import asyncio
import json
import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
# import talib  # Replaced with pandas/numpy implementations
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternType(Enum):
    GREEN_DOT_FORMATION = "green_dot_formation"
    BLOOD_DIAMOND_REVERSAL = "blood_diamond_reversal"
    YELLOW_DIAMOND_CONTINUATION = "yellow_diamond_continuation"
    ANCHOR_TRIGGER_PATTERN = "anchor_trigger_pattern"
    MONEY_FLOW_DIVERGENCE = "money_flow_divergence"
    SQUEEZE_RELEASE = "squeeze_release"
    DYNAMIC_SUPPORT_BOUNCE = "dynamic_support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    HIDDEN_BULLISH_DIVERGENCE = "hidden_bullish_divergence"
    REGULAR_BEARISH_DIVERGENCE = "regular_bearish_divergence"
    TRIPLE_CONFLUENCE_SETUP = "triple_confluence_setup"
    CROSS_TIMEFRAME_MOMENTUM = "cross_timeframe_momentum"

@dataclass
class PatternResult:
    pattern_type: PatternType
    success_rate: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timeframe: str
    market_condition: str
    volume_confirmation: bool

@dataclass
class MarketCipherSignal:
    mc_a_signal: str  # green_dot, red_x, yellow_diamond, blood_diamond
    mc_b_momentum: float  # -1 to 1
    mc_b_money_flow: float  # -1 to 1
    mc_b_vwap_momentum: float  # -1 to 1
    mc_sr_support: float
    mc_sr_resistance: float
    mc_dbsi_divergence: str  # bullish, bearish, none
    confluence_score: float  # 0 to 1

@dataclass
class LuxAlgoSignal:
    order_blocks: List[Dict]  # bullish/bearish order blocks
    market_structure: str  # bullish, bearish, neutral
    premium_discount_zone: str  # premium, discount, neutral
    smart_money_flow: str  # accumulation, distribution, neutral
    support_resistance: List[float]
    fair_value_gaps: List[Dict]

class MarketCipherAnalyzer:
    """Advanced Market Cipher Analysis with Pattern Book Integration"""
    
    def __init__(self):
        self.pattern_success_rates = {
            PatternType.GREEN_DOT_FORMATION: 0.78,
            PatternType.BLOOD_DIAMOND_REVERSAL: 0.85,
            PatternType.YELLOW_DIAMOND_CONTINUATION: 0.72,
            PatternType.ANCHOR_TRIGGER_PATTERN: 0.82,
            PatternType.MONEY_FLOW_DIVERGENCE: 0.76,
            PatternType.SQUEEZE_RELEASE: 0.74,
            PatternType.DYNAMIC_SUPPORT_BOUNCE: 0.79,
            PatternType.RESISTANCE_REJECTION: 0.77,
            PatternType.HIDDEN_BULLISH_DIVERGENCE: 0.81,
            PatternType.REGULAR_BEARISH_DIVERGENCE: 0.83,
            PatternType.TRIPLE_CONFLUENCE_SETUP: 0.89,
            PatternType.CROSS_TIMEFRAME_MOMENTUM: 0.86
        }
        
        # Optimized settings from pattern book
        self.settings = {
            'mc_a_ema_length': 21,
            'mc_a_smoothing': 3,
            'mc_b_money_flow_length': 14,
            'mc_b_momentum_length': 21,
            'mc_b_vwap_periods': 20,
            'mc_b_overbought': 80,
            'mc_b_oversold': 20,
            'mc_sr_strength': 5,
            'mc_sr_lookback': 50,
            'mc_dbsi_rsi_length': 14,
            'mc_dbsi_smoothing': 3,
            'mc_dbsi_divergence_lookback': 20
        }
    
    def analyze_market_cipher_a(self, data: pd.DataFrame) -> Dict:
        """Analyze Market Cipher A signals with optimized settings"""
        try:
            # Calculate EMAs with optimized length
            ema_fast = data['close'].ewm(span=self.settings['mc_a_ema_length']).mean()
            ema_slow = data['close'].ewm(span=self.settings['mc_a_ema_length'] * 2).mean()
            
            # EMA Ribbon analysis
            ema_ribbon_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
            ema_ribbon_strength = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / data['close'].iloc[-1]
            
            # Signal detection
            current_signal = "none"
            signal_strength = 0.0
            
            # Green Dot Formation detection
            if (ema_ribbon_bullish and 
                data['close'].iloc[-1] > ema_fast.iloc[-1] and
                data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1]):
                current_signal = "green_dot"
                signal_strength = min(0.9, 0.5 + ema_ribbon_strength * 10)
            
            # Blood Diamond detection (extreme oversold)
            rsi = self.calculate_rsi(data['close'], 14)
            if rsi.iloc[-1] < 25 and data['close'].iloc[-1] < data['low'].rolling(50).min().iloc[-1]:
                current_signal = "blood_diamond"
                signal_strength = 0.95
            
            # Yellow Diamond continuation
            elif (ema_ribbon_bullish and 
                  0.3 < rsi.iloc[-1] < 0.7 and
                  data['close'].iloc[-1] > data['high'].rolling(5).mean().iloc[-1]):
                current_signal = "yellow_diamond"
                signal_strength = 0.7
            
            return {
                'signal': current_signal,
                'strength': signal_strength,
                'ema_ribbon_bullish': ema_ribbon_bullish,
                'ema_ribbon_strength': ema_ribbon_strength,
                'rsi': rsi.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error in Market Cipher A analysis: {e}")
            return {'signal': 'none', 'strength': 0.0}
    
    def analyze_market_cipher_b(self, data: pd.DataFrame) -> Dict:
        """Analyze Market Cipher B with Anchor & Trigger Pattern focus"""
        try:
            # Money Flow calculation
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(
                self.settings['mc_b_money_flow_length']).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(
                self.settings['mc_b_money_flow_length']).sum()
            
            money_flow_ratio = positive_flow / (positive_flow + negative_flow)
            money_flow_normalized = (money_flow_ratio - 0.5) * 2  # -1 to 1
            
            # Momentum Wave calculation
            momentum = self.calculate_rsi(data['close'], self.settings['mc_b_momentum_length'])
            momentum_normalized = (momentum - 50) / 50  # -1 to 1
            
            # VWAP Momentum (Anchor & Trigger Pattern key component)
            vwap = (data['volume'] * typical_price).cumsum() / data['volume'].cumsum()
            vwap_momentum = (data['close'] - vwap) / vwap
            vwap_momentum_normalized = np.tanh(vwap_momentum * 10)  # -1 to 1
            
            # Anchor & Trigger Pattern detection
            anchor_trigger_signal = False
            if (abs(vwap_momentum_normalized.iloc[-1]) < 0.1 and  # VWAP near zero line
                momentum_normalized.iloc[-1] > -0.2 and  # Momentum turning up
                money_flow_normalized.iloc[-1] > 0.1):  # Positive money flow
                anchor_trigger_signal = True
            
            # Green Dot formation (momentum cutting in)
            green_dot_forming = (momentum_normalized.iloc[-1] > momentum_normalized.iloc[-2] and
                               money_flow_normalized.iloc[-1] > 0.2)
            
            return {
                'money_flow': money_flow_normalized.iloc[-1],
                'momentum': momentum_normalized.iloc[-1],
                'vwap_momentum': vwap_momentum_normalized.iloc[-1],
                'anchor_trigger_signal': anchor_trigger_signal,
                'green_dot_forming': green_dot_forming,
                'overbought': momentum.iloc[-1] > self.settings['mc_b_overbought'],
                'oversold': momentum.iloc[-1] < self.settings['mc_b_oversold']
            }
            
        except Exception as e:
            logger.error(f"Error in Market Cipher B analysis: {e}")
            return {'money_flow': 0.0, 'momentum': 0.0, 'vwap_momentum': 0.0}
    
    def analyze_market_cipher_sr(self, data: pd.DataFrame) -> Dict:
        """Analyze Market Cipher SR with dynamic support/resistance"""
        try:
            # Dynamic support and resistance calculation
            high_rolling = data['high'].rolling(self.settings['mc_sr_lookback']).max()
            low_rolling = data['low'].rolling(self.settings['mc_sr_lookback']).min()
            
            # Current support and resistance levels
            current_support = low_rolling.iloc[-1]
            current_resistance = high_rolling.iloc[-1]
            
            # Distance from support/resistance
            price = data['close'].iloc[-1]
            support_distance = (price - current_support) / price
            resistance_distance = (current_resistance - price) / price
            
            # Support/Resistance strength
            support_touches = sum(1 for i in range(-10, 0) 
                                if abs(data['low'].iloc[i] - current_support) / current_support < 0.01)
            resistance_touches = sum(1 for i in range(-10, 0) 
                                   if abs(data['high'].iloc[i] - current_resistance) / current_resistance < 0.01)
            
            return {
                'support_level': current_support,
                'resistance_level': current_resistance,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance,
                'support_strength': min(support_touches / 3, 1.0),
                'resistance_strength': min(resistance_touches / 3, 1.0),
                'at_support': support_distance < 0.02,
                'at_resistance': resistance_distance < 0.02
            }
            
        except Exception as e:
            logger.error(f"Error in Market Cipher SR analysis: {e}")
            return {'support_level': 0.0, 'resistance_level': 0.0}
    
    def analyze_market_cipher_dbsi(self, data: pd.DataFrame) -> Dict:
        """Analyze Market Cipher DBSI for divergences"""
        try:
            # RSI calculation
            rsi = self.calculate_rsi(data['close'], self.settings['mc_dbsi_rsi_length'])
            
            # Price peaks and troughs
            price_peaks = []
            price_troughs = []
            rsi_peaks = []
            rsi_troughs = []
            
            lookback = self.settings['mc_dbsi_divergence_lookback']
            
            for i in range(lookback, len(data)):
                # Find peaks
                if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                    data['high'].iloc[i] > data['high'].iloc[i+1] if i < len(data)-1 else True):
                    price_peaks.append((i, data['high'].iloc[i]))
                    rsi_peaks.append((i, rsi.iloc[i]))
                
                # Find troughs
                if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                    data['low'].iloc[i] < data['low'].iloc[i+1] if i < len(data)-1 else True):
                    price_troughs.append((i, data['low'].iloc[i]))
                    rsi_troughs.append((i, rsi.iloc[i]))
            
            # Divergence detection
            divergence_type = "none"
            divergence_strength = 0.0
            
            # Regular bearish divergence
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                last_price_peak = price_peaks[-1][1]
                prev_price_peak = price_peaks[-2][1]
                last_rsi_peak = rsi_peaks[-1][1]
                prev_rsi_peak = rsi_peaks[-2][1]
                
                if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
                    divergence_type = "regular_bearish"
                    divergence_strength = min(0.9, abs(last_rsi_peak - prev_rsi_peak) / 20)
            
            # Hidden bullish divergence
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                last_price_trough = price_troughs[-1][1]
                prev_price_trough = price_troughs[-2][1]
                last_rsi_trough = rsi_troughs[-1][1]
                prev_rsi_trough = rsi_troughs[-2][1]
                
                if last_price_trough > prev_price_trough and last_rsi_trough < prev_rsi_trough:
                    divergence_type = "hidden_bullish"
                    divergence_strength = min(0.9, abs(last_rsi_trough - prev_rsi_trough) / 20)
            
            return {
                'divergence_type': divergence_type,
                'divergence_strength': divergence_strength,
                'rsi_current': rsi.iloc[-1],
                'rsi_overbought': rsi.iloc[-1] > 70,
                'rsi_oversold': rsi.iloc[-1] < 30
            }
            
        except Exception as e:
            logger.error(f"Error in Market Cipher DBSI analysis: {e}")
            return {'divergence_type': 'none', 'divergence_strength': 0.0}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using pandas"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)

class LuxAlgoAnalyzer:
    """Advanced Lux Algo Price Action Analysis"""
    
    def __init__(self):
        self.settings = {
            'order_block_strength': 3,
            'market_structure_periods': 20,
            'premium_discount_threshold': 0.618,
            'smart_money_volume_threshold': 1.5
        }
    
    def analyze_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """Detect bullish and bearish order blocks"""
        try:
            order_blocks = []
            
            for i in range(20, len(data) - 5):
                # Bullish order block detection
                if (data['low'].iloc[i] < data['low'].iloc[i-1] and
                    data['close'].iloc[i] > data['open'].iloc[i] and
                    data['volume'].iloc[i] > data['volume'].rolling(10).mean().iloc[i]):
                    
                    order_blocks.append({
                        'type': 'bullish',
                        'price': data['low'].iloc[i],
                        'strength': min(1.0, data['volume'].iloc[i] / data['volume'].rolling(20).mean().iloc[i]),
                        'index': i
                    })
                
                # Bearish order block detection
                if (data['high'].iloc[i] > data['high'].iloc[i-1] and
                    data['close'].iloc[i] < data['open'].iloc[i] and
                    data['volume'].iloc[i] > data['volume'].rolling(10).mean().iloc[i]):
                    
                    order_blocks.append({
                        'type': 'bearish',
                        'price': data['high'].iloc[i],
                        'strength': min(1.0, data['volume'].iloc[i] / data['volume'].rolling(20).mean().iloc[i]),
                        'index': i
                    })
            
            # Return only recent and strong order blocks
            recent_blocks = [block for block in order_blocks if len(data) - block['index'] < 50]
            return sorted(recent_blocks, key=lambda x: x['strength'], reverse=True)[:10]
            
        except Exception as e:
            logger.error(f"Error in order block analysis: {e}")
            return []
    
    def analyze_market_structure(self, data: pd.DataFrame) -> str:
        """Analyze market structure (CHoCH/BOS)"""
        try:
            # Calculate swing highs and lows
            periods = self.settings['market_structure_periods']
            
            swing_highs = []
            swing_lows = []
            
            for i in range(periods, len(data) - periods):
                if data['high'].iloc[i] == data['high'].iloc[i-periods:i+periods].max():
                    swing_highs.append((i, data['high'].iloc[i]))
                
                if data['low'].iloc[i] == data['low'].iloc[i-periods:i+periods].min():
                    swing_lows.append((i, data['low'].iloc[i]))
            
            # Determine market structure
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                recent_highs = swing_highs[-2:]
                recent_lows = swing_lows[-2:]
                
                # Higher highs and higher lows = bullish
                if (recent_highs[-1][1] > recent_highs[-2][1] and 
                    recent_lows[-1][1] > recent_lows[-2][1]):
                    return "bullish"
                
                # Lower highs and lower lows = bearish
                elif (recent_highs[-1][1] < recent_highs[-2][1] and 
                      recent_lows[-1][1] < recent_lows[-2][1]):
                    return "bearish"
            
            return "neutral"
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return "neutral"
    
    def analyze_premium_discount_zones(self, data: pd.DataFrame) -> str:
        """Analyze premium/discount zones using Fibonacci levels"""
        try:
            # Get recent high and low
            lookback = 100
            recent_high = data['high'].iloc[-lookback:].max()
            recent_low = data['low'].iloc[-lookback:].min()
            current_price = data['close'].iloc[-1]
            
            # Calculate Fibonacci levels
            range_size = recent_high - recent_low
            fib_618 = recent_low + (range_size * 0.618)
            fib_382 = recent_low + (range_size * 0.382)
            
            # Determine zone
            if current_price > fib_618:
                return "premium"
            elif current_price < fib_382:
                return "discount"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in premium/discount analysis: {e}")
            return "neutral"
    
    def analyze_smart_money_flow(self, data: pd.DataFrame) -> str:
        """Analyze smart money concepts"""
        try:
            # Volume analysis for smart money detection
            avg_volume = data['volume'].rolling(20).mean()
            volume_spikes = data['volume'] > (avg_volume * self.settings['smart_money_volume_threshold'])
            
            # Price action during volume spikes
            recent_spikes = volume_spikes.iloc[-10:]
            spike_indices = [i for i, spike in enumerate(recent_spikes) if spike]
            
            if not spike_indices:
                return "neutral"
            
            # Analyze price movement during spikes
            accumulation_signals = 0
            distribution_signals = 0
            
            for spike_idx in spike_indices:
                actual_idx = len(data) - 10 + spike_idx
                if actual_idx < len(data) - 1:
                    # Check if price moved up after volume spike (accumulation)
                    if data['close'].iloc[actual_idx + 1] > data['close'].iloc[actual_idx]:
                        accumulation_signals += 1
                    else:
                        distribution_signals += 1
            
            if accumulation_signals > distribution_signals:
                return "accumulation"
            elif distribution_signals > accumulation_signals:
                return "distribution"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in smart money analysis: {e}")
            return "neutral"

class BubbyBotEnhancedV2:
    """Enhanced BubbyBot with comprehensive Market Cipher & Lux Algo integration"""
    
    def __init__(self):
        self.mc_analyzer = MarketCipherAnalyzer()
        self.lux_analyzer = LuxAlgoAnalyzer()
        self.setup_database()
        
        # Performance tracking
        self.pattern_performance = {}
        self.total_signals = 0
        self.successful_signals = 0
        
        logger.info("BubbyBot Enhanced V2 initialized with Pattern Book integration")
    
    def setup_database(self):
        """Setup SQLite database for pattern tracking"""
        try:
            self.conn = sqlite3.connect('bubbybot_patterns.db')
            cursor = self.conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    pattern_type TEXT,
                    success_rate REAL,
                    confidence REAL,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    timeframe TEXT,
                    market_condition TEXT,
                    actual_success BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    mc_a_signal TEXT,
                    mc_b_momentum REAL,
                    mc_sr_support REAL,
                    mc_sr_resistance REAL,
                    lux_market_structure TEXT,
                    confluence_score REAL,
                    recommendation TEXT
                )
            ''')
            
            self.conn.commit()
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    async def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Comprehensive analysis of a symbol using all systems"""
        try:
            # Get market data
            data = await self.get_market_data(symbol, timeframe)
            if data is None or len(data) < 100:
                return {'error': 'Insufficient data'}
            
            # Market Cipher Analysis
            mc_a_result = self.mc_analyzer.analyze_market_cipher_a(data)
            mc_b_result = self.mc_analyzer.analyze_market_cipher_b(data)
            mc_sr_result = self.mc_analyzer.analyze_market_cipher_sr(data)
            mc_dbsi_result = self.mc_analyzer.analyze_market_cipher_dbsi(data)
            
            # Lux Algo Analysis
            order_blocks = self.lux_analyzer.analyze_order_blocks(data)
            market_structure = self.lux_analyzer.analyze_market_structure(data)
            premium_discount = self.lux_analyzer.analyze_premium_discount_zones(data)
            smart_money = self.lux_analyzer.analyze_smart_money_flow(data)
            
            # Pattern Recognition
            detected_patterns = await self.detect_patterns(
                mc_a_result, mc_b_result, mc_sr_result, mc_dbsi_result,
                order_blocks, market_structure, premium_discount, smart_money,
                data, timeframe
            )
            
            # Confluence Scoring
            confluence_score = self.calculate_confluence_score(
                mc_a_result, mc_b_result, mc_sr_result, mc_dbsi_result,
                market_structure, premium_discount, smart_money
            )
            
            # Generate Trading Recommendation
            recommendation = await self.generate_recommendation(
                detected_patterns, confluence_score, data, timeframe
            )
            
            # Store signal in database
            await self.store_signal(symbol, mc_a_result, mc_b_result, mc_sr_result, 
                                  market_structure, confluence_score, recommendation)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'market_cipher': {
                    'mc_a': mc_a_result,
                    'mc_b': mc_b_result,
                    'mc_sr': mc_sr_result,
                    'mc_dbsi': mc_dbsi_result
                },
                'lux_algo': {
                    'order_blocks': order_blocks,
                    'market_structure': market_structure,
                    'premium_discount': premium_discount,
                    'smart_money': smart_money
                },
                'detected_patterns': detected_patterns,
                'confluence_score': confluence_score,
                'recommendation': recommendation,
                'current_price': data['close'].iloc[-1],
                'volume': data['volume'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'error': str(e)}
    
    async def detect_patterns(self, mc_a, mc_b, mc_sr, mc_dbsi, order_blocks, 
                            market_structure, premium_discount, smart_money, 
                            data, timeframe) -> List[PatternResult]:
        """Detect trading patterns based on Pattern Book"""
        patterns = []
        current_price = data['close'].iloc[-1]
        
        try:
            # Pattern 1: Green Dot Formation
            if (mc_a['signal'] == 'green_dot' and 
                mc_b['momentum'] > 0.2 and 
                mc_b['money_flow'] > 0.1):
                
                pattern = PatternResult(
                    pattern_type=PatternType.GREEN_DOT_FORMATION,
                    success_rate=self.mc_analyzer.pattern_success_rates[PatternType.GREEN_DOT_FORMATION],
                    confidence=mc_a['strength'],
                    entry_price=current_price,
                    stop_loss=mc_sr['support_level'],
                    take_profit=mc_sr['resistance_level'],
                    risk_reward_ratio=self.calculate_risk_reward(current_price, mc_sr['support_level'], mc_sr['resistance_level']),
                    timeframe=timeframe,
                    market_condition=market_structure,
                    volume_confirmation=data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1]
                )
                patterns.append(pattern)
            
            # Pattern 4: Anchor & Trigger Pattern (CryptoFace Strategy)
            if mc_b['anchor_trigger_signal']:
                pattern = PatternResult(
                    pattern_type=PatternType.ANCHOR_TRIGGER_PATTERN,
                    success_rate=self.mc_analyzer.pattern_success_rates[PatternType.ANCHOR_TRIGGER_PATTERN],
                    confidence=0.8 if mc_b['green_dot_forming'] else 0.6,
                    entry_price=current_price,
                    stop_loss=current_price * 0.98,  # 2% stop
                    take_profit=current_price * 1.06,  # 6% target
                    risk_reward_ratio=3.0,
                    timeframe=timeframe,
                    market_condition=market_structure,
                    volume_confirmation=True
                )
                patterns.append(pattern)
            
            # Pattern 2: Blood Diamond Reversal
            if mc_a['signal'] == 'blood_diamond':
                pattern = PatternResult(
                    pattern_type=PatternType.BLOOD_DIAMOND_REVERSAL,
                    success_rate=self.mc_analyzer.pattern_success_rates[PatternType.BLOOD_DIAMOND_REVERSAL],
                    confidence=0.9,
                    entry_price=current_price,
                    stop_loss=current_price * 0.95,  # 5% stop for reversal
                    take_profit=current_price * 1.15,  # 15% target
                    risk_reward_ratio=3.0,
                    timeframe=timeframe,
                    market_condition=market_structure,
                    volume_confirmation=data['volume'].iloc[-1] > data['volume'].rolling(10).mean().iloc[-1] * 1.5
                )
                patterns.append(pattern)
            
            # Pattern 11: Triple Confluence Setup
            confluence_signals = sum([
                1 if mc_a['signal'] in ['green_dot', 'yellow_diamond'] else 0,
                1 if mc_b['momentum'] > 0.1 and mc_b['money_flow'] > 0.1 else 0,
                1 if mc_sr['at_support'] or (mc_sr['support_distance'] < 0.05) else 0,
                1 if market_structure == 'bullish' else 0,
                1 if premium_discount == 'discount' else 0
            ])
            
            if confluence_signals >= 3:
                pattern = PatternResult(
                    pattern_type=PatternType.TRIPLE_CONFLUENCE_SETUP,
                    success_rate=self.mc_analyzer.pattern_success_rates[PatternType.TRIPLE_CONFLUENCE_SETUP],
                    confidence=min(0.95, confluence_signals / 5.0),
                    entry_price=current_price,
                    stop_loss=mc_sr['support_level'] * 0.99,
                    take_profit=mc_sr['resistance_level'],
                    risk_reward_ratio=self.calculate_risk_reward(current_price, mc_sr['support_level'], mc_sr['resistance_level']),
                    timeframe=timeframe,
                    market_condition=market_structure,
                    volume_confirmation=True
                )
                patterns.append(pattern)
            
            # Pattern 9: Hidden Bullish Divergence
            if mc_dbsi['divergence_type'] == 'hidden_bullish':
                pattern = PatternResult(
                    pattern_type=PatternType.HIDDEN_BULLISH_DIVERGENCE,
                    success_rate=self.mc_analyzer.pattern_success_rates[PatternType.HIDDEN_BULLISH_DIVERGENCE],
                    confidence=mc_dbsi['divergence_strength'],
                    entry_price=current_price,
                    stop_loss=current_price * 0.97,
                    take_profit=current_price * 1.08,
                    risk_reward_ratio=2.7,
                    timeframe=timeframe,
                    market_condition=market_structure,
                    volume_confirmation=True
                )
                patterns.append(pattern)
            
            # Pattern 10: Regular Bearish Divergence
            if mc_dbsi['divergence_type'] == 'regular_bearish':
                pattern = PatternResult(
                    pattern_type=PatternType.REGULAR_BEARISH_DIVERGENCE,
                    success_rate=self.mc_analyzer.pattern_success_rates[PatternType.REGULAR_BEARISH_DIVERGENCE],
                    confidence=mc_dbsi['divergence_strength'],
                    entry_price=current_price,
                    stop_loss=current_price * 1.03,  # Short position
                    take_profit=current_price * 0.92,  # 8% down target
                    risk_reward_ratio=2.7,
                    timeframe=timeframe,
                    market_condition=market_structure,
                    volume_confirmation=True
                )
                patterns.append(pattern)
            
            return sorted(patterns, key=lambda x: x.success_rate * x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def calculate_confluence_score(self, mc_a, mc_b, mc_sr, mc_dbsi, 
                                 market_structure, premium_discount, smart_money) -> float:
        """Calculate overall confluence score (0-1)"""
        try:
            score = 0.0
            max_score = 0.0
            
            # Market Cipher A contribution (25%)
            if mc_a['signal'] in ['green_dot', 'yellow_diamond']:
                score += 0.25 * mc_a['strength']
            max_score += 0.25
            
            # Market Cipher B contribution (30%)
            if mc_b['anchor_trigger_signal']:
                score += 0.15
            if mc_b['green_dot_forming']:
                score += 0.10
            if mc_b['momentum'] > 0.1:
                score += 0.05
            max_score += 0.30
            
            # Market Cipher SR contribution (15%)
            if mc_sr['at_support']:
                score += 0.10
            if mc_sr['support_strength'] > 0.5:
                score += 0.05
            max_score += 0.15
            
            # Market Cipher DBSI contribution (10%)
            if mc_dbsi['divergence_type'] in ['hidden_bullish', 'regular_bearish']:
                score += 0.10 * mc_dbsi['divergence_strength']
            max_score += 0.10
            
            # Lux Algo contributions (20%)
            if market_structure == 'bullish':
                score += 0.08
            if premium_discount == 'discount':
                score += 0.06
            if smart_money == 'accumulation':
                score += 0.06
            max_score += 0.20
            
            return min(1.0, score / max_score if max_score > 0 else 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating confluence score: {e}")
            return 0.0
    
    def calculate_risk_reward(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk/reward ratio"""
        try:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            return reward / risk if risk > 0 else 0.0
        except:
            return 0.0
    
    async def generate_recommendation(self, patterns: List[PatternResult], 
                                   confluence_score: float, data: pd.DataFrame, 
                                   timeframe: str) -> Dict:
        """Generate trading recommendation based on patterns and confluence"""
        try:
            if not patterns:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No significant patterns detected'
                }
            
            # Get best pattern
            best_pattern = patterns[0]
            
            # Determine action based on pattern and confluence
            if confluence_score > 0.7 and best_pattern.success_rate > 0.8:
                action = 'STRONG_BUY' if best_pattern.pattern_type.name.find('BEARISH') == -1 else 'STRONG_SELL'
                confidence = min(0.95, confluence_score * best_pattern.success_rate)
            elif confluence_score > 0.5 and best_pattern.success_rate > 0.7:
                action = 'BUY' if best_pattern.pattern_type.name.find('BEARISH') == -1 else 'SELL'
                confidence = confluence_score * best_pattern.success_rate
            elif confluence_score > 0.3:
                action = 'WEAK_BUY' if best_pattern.pattern_type.name.find('BEARISH') == -1 else 'WEAK_SELL'
                confidence = confluence_score * best_pattern.success_rate * 0.8
            else:
                action = 'HOLD'
                confidence = 0.3
            
            # Position sizing based on confidence
            if confidence > 0.8:
                position_size = 'LARGE'
            elif confidence > 0.6:
                position_size = 'MEDIUM'
            elif confidence > 0.4:
                position_size = 'SMALL'
            else:
                position_size = 'MINIMAL'
            
            return {
                'action': action,
                'confidence': confidence,
                'position_size': position_size,
                'primary_pattern': best_pattern.pattern_type.value,
                'pattern_success_rate': best_pattern.success_rate,
                'confluence_score': confluence_score,
                'entry_price': best_pattern.entry_price,
                'stop_loss': best_pattern.stop_loss,
                'take_profit': best_pattern.take_profit,
                'risk_reward_ratio': best_pattern.risk_reward_ratio,
                'timeframe': timeframe,
                'reason': f"Pattern: {best_pattern.pattern_type.value} | Confluence: {confluence_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Analysis error'}
    
    async def get_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get market data for analysis"""
        try:
            # Convert timeframe to yfinance format
            tf_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk'
            }
            
            yf_timeframe = tf_map.get(timeframe, '1h')
            
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1y', interval=yf_timeframe)
            
            if data.empty:
                return None
            
            # Convert column names to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def store_signal(self, symbol: str, mc_a: Dict, mc_b: Dict, mc_sr: Dict, 
                         market_structure: str, confluence_score: float, 
                         recommendation: Dict):
        """Store signal in database for tracking"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO signal_history 
                (timestamp, symbol, mc_a_signal, mc_b_momentum, mc_sr_support, 
                 mc_sr_resistance, lux_market_structure, confluence_score, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                symbol,
                mc_a.get('signal', 'none'),
                mc_b.get('momentum', 0.0),
                mc_sr.get('support_level', 0.0),
                mc_sr.get('resistance_level', 0.0),
                market_structure,
                confluence_score,
                json.dumps(recommendation)
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    async def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            cursor = self.conn.cursor()
            
            # Overall stats
            cursor.execute('SELECT COUNT(*) FROM signal_history')
            total_signals = cursor.fetchone()[0]
            
            # Pattern performance
            cursor.execute('''
                SELECT pattern_type, AVG(actual_success), COUNT(*) 
                FROM pattern_results 
                WHERE actual_success IS NOT NULL 
                GROUP BY pattern_type
            ''')
            pattern_stats = cursor.fetchall()
            
            return {
                'total_signals_generated': total_signals,
                'pattern_performance': {
                    row[0]: {'success_rate': row[1], 'count': row[2]} 
                    for row in pattern_stats
                },
                'system_uptime': 'Active',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}

# Example usage and testing
async def main():
    """Main function for testing BubbyBot Enhanced V2"""
    bot = BubbyBotEnhancedV2()
    
    # Test symbols (yfinance format)
    test_symbols = ['AVAX-USD', 'BTC-USD', 'ETH-USD']
    
    print("🤖 BubbyBot Enhanced V2 - Market Cipher & Lux Algo Analysis")
    print("=" * 60)
    
    for symbol in test_symbols:
        print(f"\n📊 Analyzing {symbol}...")
        result = await bot.analyze_symbol(symbol, '1h')
        
        if 'error' not in result:
            print(f"✅ Analysis Complete for {symbol}")
            print(f"   Confluence Score: {result['confluence_score']:.3f}")
            print(f"   Recommendation: {result['recommendation']['action']}")
            print(f"   Confidence: {result['recommendation']['confidence']:.3f}")
            print(f"   Primary Pattern: {result['recommendation'].get('primary_pattern', 'None')}")
            
            if result['detected_patterns']:
                print(f"   Detected Patterns: {len(result['detected_patterns'])}")
                for pattern in result['detected_patterns'][:3]:
                    print(f"     - {pattern.pattern_type.value}: {pattern.success_rate:.1%} success rate")
        else:
            print(f"❌ Error analyzing {symbol}: {result['error']}")
    
    # Performance stats
    print(f"\n📈 Performance Statistics:")
    stats = await bot.get_performance_stats()
    print(f"   Total Signals: {stats.get('total_signals_generated', 0)}")
    print(f"   System Status: {stats.get('system_uptime', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(main())

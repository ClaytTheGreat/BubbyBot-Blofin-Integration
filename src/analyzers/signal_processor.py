"""
Enhanced Signal Processor for Market Cipher & Lux Algo Trading Bot
Implements advanced pattern recognition and signal analysis
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal types for different trading actions"""
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NEUTRAL = "neutral"

class MarketCipherSignals(Enum):
    """Market Cipher specific signal patterns"""
    BLOOD_DIAMOND = "blood_diamond"
    YELLOW_X = "yellow_x"
    MONEY_FLOW_BULLISH = "money_flow_bullish"
    MONEY_FLOW_BEARISH = "money_flow_bearish"
    SQUEEZE_MOMENTUM = "squeeze_momentum"
    DIVERGENCE_BULLISH = "divergence_bullish"
    DIVERGENCE_BEARISH = "divergence_bearish"

class LuxAlgoSignals(Enum):
    """Lux Algo specific signal patterns"""
    ORDER_BLOCK_BULLISH = "order_block_bullish"
    ORDER_BLOCK_BEARISH = "order_block_bearish"
    PREMIUM_ZONE = "premium_zone"
    DISCOUNT_ZONE = "discount_zone"
    EQUILIBRIUM_ZONE = "equilibrium_zone"
    BREAKER_BLOCK = "breaker_block"
    LIQUIDITY_SWEEP = "liquidity_sweep"

@dataclass
class TradingSignal:
    """Enhanced trading signal with comprehensive analysis"""
    timestamp: str
    symbol: str
    timeframe: str
    signal_type: SignalType
    price: float
    confidence: float
    
    # Market Cipher components
    market_cipher_a: Dict[str, Any]
    market_cipher_b: Dict[str, Any]
    market_cipher_sr: Dict[str, Any]
    market_cipher_dbsi: Dict[str, Any]
    
    # Lux Algo components
    lux_algo_order_blocks: Dict[str, Any]
    lux_algo_premium_discount: Dict[str, Any]
    lux_algo_market_structure: Dict[str, Any]
    
    # Pattern recognition
    detected_patterns: List[str]
    confluence_score: float
    risk_reward_ratio: float
    
    # Additional metadata
    raw_data: Dict[str, Any]
    processing_notes: List[str]

class PatternRecognizer:
    """Advanced pattern recognition for Market Cipher and Lux Algo signals"""
    
    def __init__(self):
        self.market_cipher_patterns = {
            'blood_diamond': self._detect_blood_diamond,
            'yellow_x': self._detect_yellow_x,
            'money_flow_divergence': self._detect_money_flow_divergence,
            'squeeze_breakout': self._detect_squeeze_breakout,
        }
        
        self.lux_algo_patterns = {
            'order_block_confluence': self._detect_order_block_confluence,
            'premium_discount_setup': self._detect_premium_discount_setup,
            'liquidity_grab': self._detect_liquidity_grab,
            'market_structure_break': self._detect_market_structure_break,
        }
    
    def detect_patterns(self, signal_data: Dict[str, Any]) -> List[str]:
        """Detect all relevant patterns in the signal data"""
        detected = []
        
        # Market Cipher pattern detection
        for pattern_name, detector in self.market_cipher_patterns.items():
            if detector(signal_data):
                detected.append(f"mc_{pattern_name}")
        
        # Lux Algo pattern detection
        for pattern_name, detector in self.lux_algo_patterns.items():
            if detector(signal_data):
                detected.append(f"la_{pattern_name}")
        
        return detected
    
    def _detect_blood_diamond(self, data: Dict[str, Any]) -> bool:
        """Detect Market Cipher Blood Diamond pattern"""
        try:
            mc_a = data.get('market_cipher_a', {})
            return (
                mc_a.get('blood_diamond_signal', False) and
                mc_a.get('trend_strength', 0) > 0.7
            )
        except Exception as e:
            logger.error(f"Error detecting blood diamond: {e}")
            return False
    
    def _detect_yellow_x(self, data: Dict[str, Any]) -> bool:
        """Detect Market Cipher Yellow X pattern"""
        try:
            mc_a = data.get('market_cipher_a', {})
            return (
                mc_a.get('yellow_x_signal', False) and
                mc_a.get('whale_manipulation_risk', 0) > 0.6
            )
        except Exception as e:
            logger.error(f"Error detecting yellow X: {e}")
            return False
    
    def _detect_money_flow_divergence(self, data: Dict[str, Any]) -> bool:
        """Detect Market Cipher money flow divergence"""
        try:
            mc_b = data.get('market_cipher_b', {})
            return (
                mc_b.get('money_flow_divergence', False) and
                abs(mc_b.get('divergence_strength', 0)) > 0.5
            )
        except Exception as e:
            logger.error(f"Error detecting money flow divergence: {e}")
            return False
    
    def _detect_squeeze_breakout(self, data: Dict[str, Any]) -> bool:
        """Detect Market Cipher squeeze breakout"""
        try:
            mc_dbsi = data.get('market_cipher_dbsi', {})
            return (
                mc_dbsi.get('squeeze_active', False) and
                mc_dbsi.get('momentum_building', False) and
                mc_dbsi.get('breakout_imminent', False)
            )
        except Exception as e:
            logger.error(f"Error detecting squeeze breakout: {e}")
            return False
    
    def _detect_order_block_confluence(self, data: Dict[str, Any]) -> bool:
        """Detect Lux Algo order block confluence"""
        try:
            la_ob = data.get('lux_algo_order_blocks', {})
            return (
                la_ob.get('bullish_order_block', False) or
                la_ob.get('bearish_order_block', False)
            ) and la_ob.get('volume_confirmation', False)
        except Exception as e:
            logger.error(f"Error detecting order block confluence: {e}")
            return False
    
    def _detect_premium_discount_setup(self, data: Dict[str, Any]) -> bool:
        """Detect Lux Algo premium/discount zone setup"""
        try:
            la_pd = data.get('lux_algo_premium_discount', {})
            zone_type = la_pd.get('current_zone', '')
            return zone_type in ['premium', 'discount'] and la_pd.get('zone_strength', 0) > 0.6
        except Exception as e:
            logger.error(f"Error detecting premium/discount setup: {e}")
            return False
    
    def _detect_liquidity_grab(self, data: Dict[str, Any]) -> bool:
        """Detect Lux Algo liquidity grab pattern"""
        try:
            la_ms = data.get('lux_algo_market_structure', {})
            return (
                la_ms.get('liquidity_sweep', False) and
                la_ms.get('false_breakout', False)
            )
        except Exception as e:
            logger.error(f"Error detecting liquidity grab: {e}")
            return False
    
    def _detect_market_structure_break(self, data: Dict[str, Any]) -> bool:
        """Detect Lux Algo market structure break"""
        try:
            la_ms = data.get('lux_algo_market_structure', {})
            return (
                la_ms.get('structure_break', False) and
                la_ms.get('break_strength', 0) > 0.7
            )
        except Exception as e:
            logger.error(f"Error detecting market structure break: {e}")
            return False

class SignalProcessor:
    """Enhanced signal processor with advanced pattern recognition"""
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.signal_history = []
        
    def process_tradingview_signal(self, raw_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Process incoming TradingView signal with enhanced analysis"""
        try:
            # Extract basic signal information
            basic_info = self._extract_basic_info(raw_data)
            
            # Parse Market Cipher components
            mc_components = self._parse_market_cipher_components(raw_data)
            
            # Parse Lux Algo components
            la_components = self._parse_lux_algo_components(raw_data)
            
            # Detect patterns
            detected_patterns = self.pattern_recognizer.detect_patterns(raw_data)
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(mc_components, la_components, detected_patterns)
            
            # Determine signal type
            signal_type = self._determine_signal_type(mc_components, la_components, detected_patterns)
            
            # Calculate risk/reward ratio
            risk_reward = self._calculate_risk_reward_ratio(raw_data, signal_type)
            
            # Create enhanced signal
            signal = TradingSignal(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                symbol=basic_info['symbol'],
                timeframe=basic_info['timeframe'],
                signal_type=signal_type,
                price=basic_info['price'],
                confidence=confluence_score,
                market_cipher_a=mc_components['a'],
                market_cipher_b=mc_components['b'],
                market_cipher_sr=mc_components['sr'],
                market_cipher_dbsi=mc_components['dbsi'],
                lux_algo_order_blocks=la_components['order_blocks'],
                lux_algo_premium_discount=la_components['premium_discount'],
                lux_algo_market_structure=la_components['market_structure'],
                detected_patterns=detected_patterns,
                confluence_score=confluence_score,
                risk_reward_ratio=risk_reward,
                raw_data=raw_data,
                processing_notes=[]
            )
            
            # Store in history
            self.signal_history.append(signal)
            if len(self.signal_history) > 1000:  # Keep last 1000 signals
                self.signal_history.pop(0)
            
            logger.info(f"Processed enhanced signal: {signal.symbol} - {signal.signal_type.value} - "
                       f"Confidence: {signal.confidence:.2f} - Patterns: {len(signal.detected_patterns)}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing TradingView signal: {str(e)}")
            return None
    
    def _extract_basic_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic signal information"""
        return {
            'symbol': data.get('symbol', 'UNKNOWN'),
            'timeframe': data.get('timeframe', '1h'),
            'price': float(data.get('price', 0)),
            'action': data.get('action', 'unknown')
        }
    
    def _parse_market_cipher_components(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Parse Market Cipher A, B, SR, and DBSI components"""
        return {
            'a': {
                'trend_direction': data.get('mc_a_trend', 'neutral'),
                'trend_strength': data.get('mc_a_strength', 0.5),
                'blood_diamond_signal': data.get('mc_a_blood_diamond', False),
                'yellow_x_signal': data.get('mc_a_yellow_x', False),
                'whale_manipulation_risk': data.get('mc_a_whale_risk', 0),
                'ema_ribbon_color': data.get('mc_a_ema_color', 'gray')
            },
            'b': {
                'money_flow_direction': data.get('mc_b_money_flow', 'neutral'),
                'money_flow_strength': data.get('mc_b_mf_strength', 0.5),
                'wave_trend_signal': data.get('mc_b_wave_trend', 'neutral'),
                'rsi_level': data.get('mc_b_rsi', 50),
                'money_flow_divergence': data.get('mc_b_divergence', False),
                'divergence_strength': data.get('mc_b_div_strength', 0),
                'vwap_position': data.get('mc_b_vwap', 'neutral')
            },
            'sr': {
                'support_level': data.get('mc_sr_support', 0),
                'resistance_level': data.get('mc_sr_resistance', 0),
                'key_level_proximity': data.get('mc_sr_proximity', 0),
                'level_strength': data.get('mc_sr_strength', 0.5)
            },
            'dbsi': {
                'momentum_score': data.get('mc_dbsi_momentum', 0),
                'bull_bear_balance': data.get('mc_dbsi_balance', 0),
                'squeeze_active': data.get('mc_dbsi_squeeze', False),
                'momentum_building': data.get('mc_dbsi_building', False),
                'breakout_imminent': data.get('mc_dbsi_breakout', False),
                'dynamic_ma_direction': data.get('mc_dbsi_ma_dir', 'neutral')
            }
        }
    
    def _parse_lux_algo_components(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Parse Lux Algo order blocks, premium/discount, and market structure"""
        return {
            'order_blocks': {
                'bullish_order_block': data.get('la_ob_bullish', False),
                'bearish_order_block': data.get('la_ob_bearish', False),
                'order_block_strength': data.get('la_ob_strength', 0),
                'volume_confirmation': data.get('la_ob_volume', False),
                'breaker_block_active': data.get('la_ob_breaker', False),
                'mitigation_method': data.get('la_ob_mitigation', 'close')
            },
            'premium_discount': {
                'current_zone': data.get('la_pd_zone', 'equilibrium'),
                'zone_strength': data.get('la_pd_strength', 0.5),
                'premium_level': data.get('la_pd_premium', 0),
                'discount_level': data.get('la_pd_discount', 0),
                'equilibrium_level': data.get('la_pd_equilibrium', 0)
            },
            'market_structure': {
                'structure_break': data.get('la_ms_break', False),
                'break_direction': data.get('la_ms_direction', 'neutral'),
                'break_strength': data.get('la_ms_strength', 0),
                'liquidity_sweep': data.get('la_ms_liquidity', False),
                'false_breakout': data.get('la_ms_false_break', False),
                'higher_high': data.get('la_ms_hh', False),
                'higher_low': data.get('la_ms_hl', False),
                'lower_high': data.get('la_ms_lh', False),
                'lower_low': data.get('la_ms_ll', False)
            }
        }
    
    def _calculate_confluence_score(self, mc_components: Dict, la_components: Dict, patterns: List[str]) -> float:
        """Calculate advanced confluence score based on all components"""
        score = 0.0
        max_score = 0.0
        
        # Market Cipher A contribution (20%)
        mc_a = mc_components['a']
        if mc_a['trend_direction'] != 'neutral':
            score += mc_a['trend_strength'] * 0.2
        max_score += 0.2
        
        # Market Cipher B contribution (25%)
        mc_b = mc_components['b']
        if mc_b['money_flow_direction'] != 'neutral':
            score += mc_b['money_flow_strength'] * 0.25
        max_score += 0.25
        
        # Market Cipher DBSI contribution (15%)
        mc_dbsi = mc_components['dbsi']
        if abs(mc_dbsi['momentum_score']) > 0.3:
            score += min(abs(mc_dbsi['momentum_score']), 1.0) * 0.15
        max_score += 0.15
        
        # Lux Algo Order Blocks contribution (20%)
        la_ob = la_components['order_blocks']
        if la_ob['bullish_order_block'] or la_ob['bearish_order_block']:
            score += la_ob['order_block_strength'] * 0.2
        max_score += 0.2
        
        # Lux Algo Premium/Discount contribution (10%)
        la_pd = la_components['premium_discount']
        if la_pd['current_zone'] in ['premium', 'discount']:
            score += la_pd['zone_strength'] * 0.1
        max_score += 0.1
        
        # Pattern bonus (10%)
        pattern_bonus = min(len(patterns) * 0.02, 0.1)
        score += pattern_bonus
        max_score += 0.1
        
        # Normalize score
        if max_score > 0:
            normalized_score = score / max_score
        else:
            normalized_score = 0.5
        
        return min(max(normalized_score, 0.0), 1.0)
    
    def _determine_signal_type(self, mc_components: Dict, la_components: Dict, patterns: List[str]) -> SignalType:
        """Determine the signal type based on all available information"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Market Cipher A analysis
        mc_a = mc_components['a']
        if mc_a['trend_direction'] == 'bullish':
            bullish_signals += 1
        elif mc_a['trend_direction'] == 'bearish':
            bearish_signals += 1
        
        # Market Cipher B analysis
        mc_b = mc_components['b']
        if mc_b['money_flow_direction'] == 'bullish':
            bullish_signals += 1
        elif mc_b['money_flow_direction'] == 'bearish':
            bearish_signals += 1
        
        # Lux Algo Order Blocks analysis
        la_ob = la_components['order_blocks']
        if la_ob['bullish_order_block']:
            bullish_signals += 1
        elif la_ob['bearish_order_block']:
            bearish_signals += 1
        
        # Premium/Discount zone analysis
        la_pd = la_components['premium_discount']
        if la_pd['current_zone'] == 'discount':
            bullish_signals += 0.5  # Slight bullish bias in discount zone
        elif la_pd['current_zone'] == 'premium':
            bearish_signals += 0.5  # Slight bearish bias in premium zone
        
        # Pattern analysis
        for pattern in patterns:
            if any(bullish_term in pattern.lower() for bullish_term in ['bullish', 'long', 'buy']):
                bullish_signals += 0.5
            elif any(bearish_term in pattern.lower() for bearish_term in ['bearish', 'short', 'sell']):
                bearish_signals += 0.5
        
        # Determine final signal
        if bullish_signals > bearish_signals + 0.5:
            return SignalType.LONG
        elif bearish_signals > bullish_signals + 0.5:
            return SignalType.SHORT
        else:
            return SignalType.NEUTRAL
    
    def _calculate_risk_reward_ratio(self, data: Dict[str, Any], signal_type: SignalType) -> float:
        """Calculate risk/reward ratio based on support/resistance levels"""
        try:
            price = float(data.get('price', 0))
            support = float(data.get('mc_sr_support', price * 0.98))
            resistance = float(data.get('mc_sr_resistance', price * 1.02))
            
            if signal_type == SignalType.LONG:
                risk = abs(price - support)
                reward = abs(resistance - price)
            elif signal_type == SignalType.SHORT:
                risk = abs(resistance - price)
                reward = abs(price - support)
            else:
                return 1.0
            
            if risk > 0:
                return reward / risk
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating risk/reward ratio: {e}")
            return 1.0
    
    def get_signal_history(self, limit: int = 50) -> List[TradingSignal]:
        """Get recent signal history"""
        return self.signal_history[-limit:]
    
    def get_pattern_statistics(self) -> Dict[str, int]:
        """Get statistics on detected patterns"""
        pattern_counts = {}
        for signal in self.signal_history:
            for pattern in signal.detected_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        return pattern_counts

# Global signal processor instance
signal_processor = SignalProcessor()

def process_tradingview_signal(data: Dict[str, Any]) -> Optional[TradingSignal]:
    """Main function to process TradingView signals"""
    return signal_processor.process_tradingview_signal(data)

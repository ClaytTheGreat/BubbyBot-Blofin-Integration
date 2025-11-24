"""
Advanced Confluence Engine for Market Cipher & Lux Algo Trading Bot
Implements sophisticated confluence scoring and multi-timeframe analysis
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TimeframeWeight(Enum):
    """Timeframe weights for confluence calculation"""
    M1 = 0.05   # 1 minute
    M5 = 0.10   # 5 minutes
    M15 = 0.15  # 15 minutes
    M30 = 0.20  # 30 minutes
    H1 = 0.25   # 1 hour
    H4 = 0.30   # 4 hours
    D1 = 0.35   # Daily
    W1 = 0.40   # Weekly

class ConfluenceLevel(Enum):
    """Confluence strength levels"""
    VERY_LOW = (0.0, 0.2)
    LOW = (0.2, 0.4)
    MEDIUM = (0.4, 0.6)
    HIGH = (0.6, 0.8)
    VERY_HIGH = (0.8, 1.0)

@dataclass
class ConfluenceComponent:
    """Individual confluence component"""
    name: str
    value: float
    weight: float
    timeframe: str
    confidence: float
    description: str

@dataclass
class ConfluenceResult:
    """Complete confluence analysis result"""
    overall_score: float
    confidence_level: ConfluenceLevel
    components: List[ConfluenceComponent]
    timeframe_scores: Dict[str, float]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

class MarketCipherConfluence:
    """Market Cipher specific confluence calculations"""
    
    @staticmethod
    def calculate_trend_confluence(mc_a_data: Dict[str, Any]) -> float:
        """Calculate Market Cipher A trend confluence"""
        try:
            trend_strength = mc_a_data.get('trend_strength', 0.5)
            ema_alignment = mc_a_data.get('ema_alignment', 0.5)
            blood_diamond = 1.0 if mc_a_data.get('blood_diamond_signal', False) else 0.0
            yellow_x_penalty = -0.3 if mc_a_data.get('yellow_x_signal', False) else 0.0
            
            # Weighted combination
            confluence = (
                trend_strength * 0.4 +
                ema_alignment * 0.3 +
                blood_diamond * 0.2 +
                yellow_x_penalty * 0.1
            )
            
            return max(0.0, min(1.0, confluence))
            
        except Exception as e:
            logger.error(f"Error calculating MC-A confluence: {e}")
            return 0.5
    
    @staticmethod
    def calculate_momentum_confluence(mc_b_data: Dict[str, Any]) -> float:
        """Calculate Market Cipher B momentum confluence"""
        try:
            money_flow_strength = mc_b_data.get('money_flow_strength', 0.5)
            wave_trend_alignment = mc_b_data.get('wave_trend_alignment', 0.5)
            rsi_level = mc_b_data.get('rsi_level', 50)
            divergence_bonus = 0.2 if mc_b_data.get('money_flow_divergence', False) else 0.0
            
            # RSI normalization (extreme levels get higher scores)
            rsi_score = 1.0 - abs(rsi_level - 50) / 50.0
            if rsi_level < 30 or rsi_level > 70:
                rsi_score += 0.2  # Bonus for extreme levels
            
            confluence = (
                money_flow_strength * 0.35 +
                wave_trend_alignment * 0.25 +
                rsi_score * 0.25 +
                divergence_bonus * 0.15
            )
            
            return max(0.0, min(1.0, confluence))
            
        except Exception as e:
            logger.error(f"Error calculating MC-B confluence: {e}")
            return 0.5
    
    @staticmethod
    def calculate_support_resistance_confluence(mc_sr_data: Dict[str, Any], current_price: float) -> float:
        """Calculate Market Cipher SR confluence"""
        try:
            support_level = mc_sr_data.get('support_level', current_price * 0.98)
            resistance_level = mc_sr_data.get('resistance_level', current_price * 1.02)
            level_strength = mc_sr_data.get('level_strength', 0.5)
            
            # Calculate proximity to key levels
            support_distance = abs(current_price - support_level) / current_price
            resistance_distance = abs(current_price - resistance_level) / current_price
            
            # Closer to key levels = higher confluence
            proximity_score = 1.0 - min(support_distance, resistance_distance) * 10
            proximity_score = max(0.0, min(1.0, proximity_score))
            
            confluence = (level_strength * 0.6) + (proximity_score * 0.4)
            
            return max(0.0, min(1.0, confluence))
            
        except Exception as e:
            logger.error(f"Error calculating MC-SR confluence: {e}")
            return 0.5
    
    @staticmethod
    def calculate_dbsi_confluence(mc_dbsi_data: Dict[str, Any]) -> float:
        """Calculate Market Cipher DBSI confluence"""
        try:
            momentum_score = abs(mc_dbsi_data.get('momentum_score', 0))
            bull_bear_balance = abs(mc_dbsi_data.get('bull_bear_balance', 0))
            squeeze_bonus = 0.3 if mc_dbsi_data.get('squeeze_active', False) else 0.0
            breakout_bonus = 0.2 if mc_dbsi_data.get('breakout_imminent', False) else 0.0
            
            confluence = (
                momentum_score * 0.4 +
                bull_bear_balance * 0.3 +
                squeeze_bonus * 0.2 +
                breakout_bonus * 0.1
            )
            
            return max(0.0, min(1.0, confluence))
            
        except Exception as e:
            logger.error(f"Error calculating MC-DBSI confluence: {e}")
            return 0.5

class LuxAlgoConfluence:
    """Lux Algo specific confluence calculations"""
    
    @staticmethod
    def calculate_order_block_confluence(la_ob_data: Dict[str, Any]) -> float:
        """Calculate Lux Algo order block confluence"""
        try:
            has_order_block = (
                la_ob_data.get('bullish_order_block', False) or
                la_ob_data.get('bearish_order_block', False)
            )
            
            if not has_order_block:
                return 0.0
            
            order_block_strength = la_ob_data.get('order_block_strength', 0.5)
            volume_confirmation = 0.3 if la_ob_data.get('volume_confirmation', False) else 0.0
            breaker_block_bonus = 0.2 if la_ob_data.get('breaker_block_active', False) else 0.0
            
            confluence = (
                order_block_strength * 0.5 +
                volume_confirmation * 0.3 +
                breaker_block_bonus * 0.2
            )
            
            return max(0.0, min(1.0, confluence))
            
        except Exception as e:
            logger.error(f"Error calculating LA-OB confluence: {e}")
            return 0.5
    
    @staticmethod
    def calculate_premium_discount_confluence(la_pd_data: Dict[str, Any]) -> float:
        """Calculate Lux Algo premium/discount confluence"""
        try:
            current_zone = la_pd_data.get('current_zone', 'equilibrium')
            zone_strength = la_pd_data.get('zone_strength', 0.5)
            
            # Premium and discount zones get higher scores than equilibrium
            zone_multiplier = {
                'premium': 1.0,
                'discount': 1.0,
                'equilibrium': 0.3
            }.get(current_zone, 0.5)
            
            confluence = zone_strength * zone_multiplier
            
            return max(0.0, min(1.0, confluence))
            
        except Exception as e:
            logger.error(f"Error calculating LA-PD confluence: {e}")
            return 0.5
    
    @staticmethod
    def calculate_market_structure_confluence(la_ms_data: Dict[str, Any]) -> float:
        """Calculate Lux Algo market structure confluence"""
        try:
            structure_break = 0.4 if la_ms_data.get('structure_break', False) else 0.0
            break_strength = la_ms_data.get('break_strength', 0) * 0.3
            liquidity_sweep_bonus = 0.2 if la_ms_data.get('liquidity_sweep', False) else 0.0
            false_breakout_penalty = -0.3 if la_ms_data.get('false_breakout', False) else 0.0
            
            # Higher highs/lows pattern recognition
            hh_hl_pattern = (
                la_ms_data.get('higher_high', False) and
                la_ms_data.get('higher_low', False)
            )
            lh_ll_pattern = (
                la_ms_data.get('lower_high', False) and
                la_ms_data.get('lower_low', False)
            )
            
            pattern_bonus = 0.1 if (hh_hl_pattern or lh_ll_pattern) else 0.0
            
            confluence = (
                structure_break +
                break_strength +
                liquidity_sweep_bonus +
                false_breakout_penalty +
                pattern_bonus
            )
            
            return max(0.0, min(1.0, confluence))
            
        except Exception as e:
            logger.error(f"Error calculating LA-MS confluence: {e}")
            return 0.5

class AdvancedConfluenceEngine:
    """Advanced confluence engine with multi-timeframe analysis"""
    
    def __init__(self):
        self.mc_confluence = MarketCipherConfluence()
        self.la_confluence = LuxAlgoConfluence()
        self.confluence_history = []
        
    def calculate_comprehensive_confluence(self, signal_data: Dict[str, Any]) -> ConfluenceResult:
        """Calculate comprehensive confluence score with detailed analysis"""
        try:
            components = []
            timeframe_scores = {}
            current_price = signal_data.get('price', 0)
            timeframe = signal_data.get('timeframe', '1h')
            
            # Market Cipher components
            mc_a_score = self.mc_confluence.calculate_trend_confluence(
                signal_data.get('market_cipher_a', {})
            )
            components.append(ConfluenceComponent(
                name="MC-A Trend",
                value=mc_a_score,
                weight=0.20,
                timeframe=timeframe,
                confidence=0.85,
                description="Market Cipher A trend analysis and EMA ribbon alignment"
            ))
            
            mc_b_score = self.mc_confluence.calculate_momentum_confluence(
                signal_data.get('market_cipher_b', {})
            )
            components.append(ConfluenceComponent(
                name="MC-B Momentum",
                value=mc_b_score,
                weight=0.25,
                timeframe=timeframe,
                confidence=0.80,
                description="Market Cipher B money flow and momentum indicators"
            ))
            
            mc_sr_score = self.mc_confluence.calculate_support_resistance_confluence(
                signal_data.get('market_cipher_sr', {}), current_price
            )
            components.append(ConfluenceComponent(
                name="MC-SR Levels",
                value=mc_sr_score,
                weight=0.15,
                timeframe=timeframe,
                confidence=0.75,
                description="Market Cipher support and resistance level analysis"
            ))
            
            mc_dbsi_score = self.mc_confluence.calculate_dbsi_confluence(
                signal_data.get('market_cipher_dbsi', {})
            )
            components.append(ConfluenceComponent(
                name="MC-DBSI",
                value=mc_dbsi_score,
                weight=0.15,
                timeframe=timeframe,
                confidence=0.90,
                description="Market Cipher Dual Band Strength Index momentum"
            ))
            
            # Lux Algo components
            la_ob_score = self.la_confluence.calculate_order_block_confluence(
                signal_data.get('lux_algo_order_blocks', {})
            )
            components.append(ConfluenceComponent(
                name="LA Order Blocks",
                value=la_ob_score,
                weight=0.15,
                timeframe=timeframe,
                confidence=0.85,
                description="Lux Algo volumetric order blocks and institutional levels"
            ))
            
            la_pd_score = self.la_confluence.calculate_premium_discount_confluence(
                signal_data.get('lux_algo_premium_discount', {})
            )
            components.append(ConfluenceComponent(
                name="LA Premium/Discount",
                value=la_pd_score,
                weight=0.05,
                timeframe=timeframe,
                confidence=0.70,
                description="Lux Algo premium and discount zone analysis"
            ))
            
            la_ms_score = self.la_confluence.calculate_market_structure_confluence(
                signal_data.get('lux_algo_market_structure', {})
            )
            components.append(ConfluenceComponent(
                name="LA Market Structure",
                value=la_ms_score,
                weight=0.05,
                timeframe=timeframe,
                confidence=0.75,
                description="Lux Algo market structure and liquidity analysis"
            ))
            
            # Calculate overall score
            overall_score = self._calculate_weighted_score(components)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_score)
            
            # Calculate timeframe scores
            timeframe_scores[timeframe] = overall_score
            
            # Risk assessment
            risk_assessment = self._assess_risk(signal_data, overall_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(components, overall_score, risk_assessment)
            
            result = ConfluenceResult(
                overall_score=overall_score,
                confidence_level=confidence_level,
                components=components,
                timeframe_scores=timeframe_scores,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
            # Store in history
            self.confluence_history.append(result)
            if len(self.confluence_history) > 500:  # Keep last 500 results
                self.confluence_history.pop(0)
            
            logger.info(f"Calculated confluence: {overall_score:.3f} ({confidence_level.name})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive confluence: {e}")
            return self._create_default_result()
    
    def _calculate_weighted_score(self, components: List[ConfluenceComponent]) -> float:
        """Calculate weighted confluence score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for component in components:
            weighted_score = component.value * component.weight * component.confidence
            total_weighted_score += weighted_score
            total_weight += component.weight * component.confidence
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.5
    
    def _determine_confidence_level(self, score: float) -> ConfluenceLevel:
        """Determine confidence level based on score"""
        for level in ConfluenceLevel:
            min_val, max_val = level.value
            if min_val <= score < max_val:
                return level
        return ConfluenceLevel.VERY_HIGH  # For score = 1.0
    
    def _assess_risk(self, signal_data: Dict[str, Any], confluence_score: float) -> Dict[str, Any]:
        """Assess risk factors for the signal"""
        risk_factors = []
        risk_score = 0.0
        
        # Market volatility assessment
        volatility = signal_data.get('volatility', 0.5)
        if volatility > 0.7:
            risk_factors.append("High market volatility detected")
            risk_score += 0.3
        
        # Yellow X warning from Market Cipher A
        if signal_data.get('market_cipher_a', {}).get('yellow_x_signal', False):
            risk_factors.append("Market Cipher Yellow X warning - whale manipulation risk")
            risk_score += 0.4
        
        # False breakout risk from Lux Algo
        if signal_data.get('lux_algo_market_structure', {}).get('false_breakout', False):
            risk_factors.append("Potential false breakout detected")
            risk_score += 0.3
        
        # Low confluence penalty
        if confluence_score < 0.4:
            risk_factors.append("Low confluence score increases uncertainty")
            risk_score += 0.2
        
        # Calculate position sizing recommendation
        if confluence_score > 0.8:
            position_size_multiplier = 1.5
        elif confluence_score > 0.6:
            position_size_multiplier = 1.0
        elif confluence_score > 0.4:
            position_size_multiplier = 0.5
        else:
            position_size_multiplier = 0.25
        
        return {
            'risk_factors': risk_factors,
            'risk_score': min(risk_score, 1.0),
            'position_size_multiplier': position_size_multiplier,
            'recommended_stop_loss': self._calculate_stop_loss(signal_data),
            'recommended_take_profit': self._calculate_take_profit(signal_data, confluence_score)
        }
    
    def _calculate_stop_loss(self, signal_data: Dict[str, Any]) -> float:
        """Calculate recommended stop loss level"""
        try:
            current_price = signal_data.get('price', 0)
            support_level = signal_data.get('market_cipher_sr', {}).get('support_level', current_price * 0.98)
            resistance_level = signal_data.get('market_cipher_sr', {}).get('resistance_level', current_price * 1.02)
            
            # For long positions, stop loss below support
            # For short positions, stop loss above resistance
            signal_type = signal_data.get('signal_type', 'neutral')
            
            if signal_type == 'long':
                return support_level * 0.995  # 0.5% below support
            elif signal_type == 'short':
                return resistance_level * 1.005  # 0.5% above resistance
            else:
                return current_price * 0.98  # Default 2% stop loss
                
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return signal_data.get('price', 0) * 0.98
    
    def _calculate_take_profit(self, signal_data: Dict[str, Any], confluence_score: float) -> List[float]:
        """Calculate recommended take profit levels"""
        try:
            current_price = signal_data.get('price', 0)
            resistance_level = signal_data.get('market_cipher_sr', {}).get('resistance_level', current_price * 1.02)
            support_level = signal_data.get('market_cipher_sr', {}).get('support_level', current_price * 0.98)
            
            signal_type = signal_data.get('signal_type', 'neutral')
            
            # Adjust targets based on confluence score
            multiplier = 1.0 + (confluence_score - 0.5)  # 0.5 to 1.5 range
            
            if signal_type == 'long':
                tp1 = current_price + (resistance_level - current_price) * 0.5 * multiplier
                tp2 = current_price + (resistance_level - current_price) * 1.0 * multiplier
                tp3 = current_price + (resistance_level - current_price) * 1.618 * multiplier  # Fibonacci extension
                return [tp1, tp2, tp3]
            elif signal_type == 'short':
                tp1 = current_price - (current_price - support_level) * 0.5 * multiplier
                tp2 = current_price - (current_price - support_level) * 1.0 * multiplier
                tp3 = current_price - (current_price - support_level) * 1.618 * multiplier
                return [tp1, tp2, tp3]
            else:
                return [current_price * 1.01, current_price * 1.02, current_price * 1.03]
                
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            current_price = signal_data.get('price', 0)
            return [current_price * 1.01, current_price * 1.02, current_price * 1.03]
    
    def _generate_recommendations(self, components: List[ConfluenceComponent], 
                                overall_score: float, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on confluence analysis"""
        recommendations = []
        
        # Overall score recommendations
        if overall_score >= 0.8:
            recommendations.append("HIGH CONFIDENCE: Strong confluence across multiple indicators")
            recommendations.append("Consider increasing position size within risk management limits")
        elif overall_score >= 0.6:
            recommendations.append("MEDIUM-HIGH CONFIDENCE: Good confluence alignment")
            recommendations.append("Standard position sizing recommended")
        elif overall_score >= 0.4:
            recommendations.append("MEDIUM CONFIDENCE: Mixed signals detected")
            recommendations.append("Reduce position size and wait for better confluence")
        else:
            recommendations.append("LOW CONFIDENCE: Weak confluence signals")
            recommendations.append("Consider avoiding this trade or using minimal position size")
        
        # Component-specific recommendations
        weak_components = [c for c in components if c.value < 0.3]
        if weak_components:
            weak_names = [c.name for c in weak_components]
            recommendations.append(f"Weak signals from: {', '.join(weak_names)}")
        
        strong_components = [c for c in components if c.value > 0.8]
        if strong_components:
            strong_names = [c.name for c in strong_components]
            recommendations.append(f"Strong signals from: {', '.join(strong_names)}")
        
        # Risk-based recommendations
        if risk_assessment['risk_score'] > 0.6:
            recommendations.append("HIGH RISK: Multiple risk factors detected - exercise caution")
        
        if risk_assessment['risk_factors']:
            recommendations.append("Risk factors to monitor:")
            recommendations.extend([f"  - {factor}" for factor in risk_assessment['risk_factors']])
        
        return recommendations
    
    def _create_default_result(self) -> ConfluenceResult:
        """Create default confluence result for error cases"""
        return ConfluenceResult(
            overall_score=0.5,
            confidence_level=ConfluenceLevel.MEDIUM,
            components=[],
            timeframe_scores={},
            risk_assessment={'risk_factors': [], 'risk_score': 0.5, 'position_size_multiplier': 0.5},
            recommendations=["Error in confluence calculation - use caution"],
            timestamp=datetime.now().isoformat()
        )
    
    def get_confluence_history(self, limit: int = 50) -> List[ConfluenceResult]:
        """Get recent confluence calculation history"""
        return self.confluence_history[-limit:]
    
    def get_average_confluence_score(self, timeframe_hours: int = 24) -> float:
        """Get average confluence score over specified timeframe"""
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        recent_results = [
            result for result in self.confluence_history
            if datetime.fromisoformat(result.timestamp) > cutoff_time
        ]
        
        if recent_results:
            return sum(result.overall_score for result in recent_results) / len(recent_results)
        else:
            return 0.5

# Global confluence engine instance
confluence_engine = AdvancedConfluenceEngine()

def calculate_confluence_score(signal_data: Dict[str, Any]) -> ConfluenceResult:
    """Main function to calculate confluence score"""
    return confluence_engine.calculate_comprehensive_confluence(signal_data)

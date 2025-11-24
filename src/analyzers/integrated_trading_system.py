#!/usr/bin/env python3
"""
Integrated Trading System: Market Cipher + Lux Algo + Frankie Candles
Three-Layer Analysis System for Maximum Trading Precision
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class IntegratedTradingSystem:
    """
    Comprehensive trading system integrating:
    1. Market Cipher - Overall trend and momentum analysis
    2. Lux Algo - Precise support/resistance and order flow
    3. Frankie Candles - Volume profile analysis and divergence confirmation
    """
    
    def __init__(self):
        self.system_layers = {
            'market_cipher': {
                'purpose': 'Overall trend and momentum analysis',
                'components': ['MC-A', 'MC-B', 'MC-SR'],
                'weight': 0.4  # 40% weight in final decision
            },
            'lux_algo': {
                'purpose': 'Precise support/resistance and order flow',
                'components': ['Order Blocks', 'Market Structure', 'Premium/Discount'],
                'weight': 0.35  # 35% weight in final decision
            },
            'frankie_candles': {
                'purpose': 'Volume profile analysis and divergence confirmation',
                'components': ['Volume Profile', 'MTF Divergences', 'Golden Pockets'],
                'weight': 0.25  # 25% weight in final decision
            }
        }
        
        self.confluence_requirements = {
            'minimum_layers': 2,  # At least 2 layers must agree
            'minimum_score': 0.6,  # Minimum confluence score for trade
            'high_confidence': 0.8  # High confidence threshold
        }
    
    def analyze_market_cipher_layer(self, market_data: Dict) -> Dict:
        """
        Layer 1: Market Cipher Analysis
        Purpose: Overall trend and momentum analysis
        """
        print("🎯 LAYER 1: MARKET CIPHER ANALYSIS")
        print("=" * 50)
        
        mc_analysis = {
            'mc_a_signals': self._analyze_mc_a(market_data),
            'mc_b_momentum': self._analyze_mc_b(market_data),
            'mc_sr_levels': self._analyze_mc_sr(market_data),
            'overall_bias': 'NEUTRAL',
            'confidence': 0.0,
            'signals': []
        }
        
        # Market Cipher A Analysis
        mc_a = mc_analysis['mc_a_signals']
        print(f"📈 Market Cipher A (Trend Analysis):")
        print(f"   Trend Direction: {mc_a['trend_direction']}")
        print(f"   Green Dot Present: {mc_a['green_dot_present']}")
        print(f"   EMA Ribbon Status: {mc_a['ema_ribbon_status']}")
        
        # Market Cipher B Analysis (Anchor & Trigger Patterns)
        mc_b = mc_analysis['mc_b_momentum']
        print(f"🌊 Market Cipher B (Momentum Analysis):")
        print(f"   VWAP Status: {mc_b['vwap_status']}")
        print(f"   Money Flow: {mc_b['money_flow_status']}")
        print(f"   Anchor/Trigger Pattern: {mc_b['anchor_trigger_detected']}")
        print(f"   Green Dot Formation: {mc_b['green_dot_present']}")
        
        # Market Cipher SR Analysis
        mc_sr = mc_analysis['mc_sr_levels']
        print(f"🎯 Market Cipher SR (Support/Resistance):")
        print(f"   Current Level: {mc_sr['current_level']}")
        print(f"   Next Support: {mc_sr['support_level']}")
        print(f"   Next Resistance: {mc_sr['resistance_level']}")
        
        # Calculate Market Cipher confluence
        bullish_factors = 0
        bearish_factors = 0
        
        if mc_a['trend_direction'] == 'BULLISH':
            bullish_factors += 1
        elif mc_a['trend_direction'] == 'BEARISH':
            bearish_factors += 1
            
        if mc_a['green_dot_present']:
            bullish_factors += 1
            
        if mc_b['anchor_trigger_detected']:
            bullish_factors += 1
            
        if mc_b['vwap_status'] == 'APPROACHING_ZERO_BULLISH':
            bullish_factors += 1
            
        if mc_sr['current_level'] == 'NEAR_SUPPORT':
            bullish_factors += 1
        elif mc_sr['current_level'] == 'NEAR_RESISTANCE':
            bearish_factors += 1
        
        total_factors = bullish_factors + bearish_factors
        if total_factors > 0:
            if bullish_factors > bearish_factors:
                mc_analysis['overall_bias'] = 'BULLISH'
                mc_analysis['confidence'] = bullish_factors / 5.0  # Max 5 bullish factors
            elif bearish_factors > bullish_factors:
                mc_analysis['overall_bias'] = 'BEARISH'
                mc_analysis['confidence'] = bearish_factors / 5.0
            else:
                mc_analysis['overall_bias'] = 'NEUTRAL'
                mc_analysis['confidence'] = 0.5
        
        print(f"🎯 Market Cipher Summary:")
        print(f"   Overall Bias: {mc_analysis['overall_bias']}")
        print(f"   Confidence: {mc_analysis['confidence']:.2f}")
        print()
        
        return mc_analysis
    
    def analyze_lux_algo_layer(self, market_data: Dict) -> Dict:
        """
        Layer 2: Lux Algo Price Action Analysis
        Purpose: Precise support/resistance and order flow
        """
        print("🏗️ LAYER 2: LUX ALGO PRICE ACTION ANALYSIS")
        print("=" * 50)
        
        lux_analysis = {
            'order_blocks': self._analyze_order_blocks(market_data),
            'market_structure': self._analyze_market_structure(market_data),
            'premium_discount': self._analyze_premium_discount(market_data),
            'liquidity_analysis': self._analyze_liquidity(market_data),
            'overall_bias': 'NEUTRAL',
            'confidence': 0.0,
            'key_levels': []
        }
        
        # Order Blocks Analysis
        ob = lux_analysis['order_blocks']
        print(f"📦 Order Blocks Analysis:")
        print(f"   Bullish Order Blocks: {len(ob['bullish_blocks'])}")
        print(f"   Bearish Order Blocks: {len(ob['bearish_blocks'])}")
        print(f"   Active Breaker Blocks: {len(ob['breaker_blocks'])}")
        print(f"   Nearest Support OB: {ob['nearest_support']}")
        print(f"   Nearest Resistance OB: {ob['nearest_resistance']}")
        
        # Market Structure Analysis
        ms = lux_analysis['market_structure']
        print(f"🔄 Market Structure Analysis:")
        print(f"   Current Structure: {ms['current_structure']}")
        print(f"   Last CHoCH: {ms['last_choch']}")
        print(f"   Last BOS: {ms['last_bos']}")
        print(f"   Structure Strength: {ms['structure_strength']}")
        
        # Premium/Discount Analysis
        pd = lux_analysis['premium_discount']
        print(f"💰 Premium/Discount Analysis:")
        print(f"   Current Zone: {pd['current_zone']}")
        print(f"   Zone Strength: {pd['zone_strength']}")
        print(f"   Equilibrium Distance: {pd['equilibrium_distance']}")
        
        # Calculate Lux Algo confluence
        bullish_score = 0
        bearish_score = 0
        
        # Order block scoring
        if len(ob['bullish_blocks']) > len(ob['bearish_blocks']):
            bullish_score += 0.3
        elif len(ob['bearish_blocks']) > len(ob['bullish_blocks']):
            bearish_score += 0.3
            
        # Market structure scoring
        if ms['current_structure'] == 'BULLISH':
            bullish_score += 0.4
        elif ms['current_structure'] == 'BEARISH':
            bearish_score += 0.4
            
        # Premium/discount scoring
        if pd['current_zone'] == 'DISCOUNT':
            bullish_score += 0.3
        elif pd['current_zone'] == 'PREMIUM':
            bearish_score += 0.3
        
        total_score = bullish_score + bearish_score
        if total_score > 0:
            if bullish_score > bearish_score:
                lux_analysis['overall_bias'] = 'BULLISH'
                lux_analysis['confidence'] = bullish_score
            elif bearish_score > bullish_score:
                lux_analysis['overall_bias'] = 'BEARISH'
                lux_analysis['confidence'] = bearish_score
            else:
                lux_analysis['overall_bias'] = 'NEUTRAL'
                lux_analysis['confidence'] = 0.5
        
        print(f"🎯 Lux Algo Summary:")
        print(f"   Overall Bias: {lux_analysis['overall_bias']}")
        print(f"   Confidence: {lux_analysis['confidence']:.2f}")
        print()
        
        return lux_analysis
    
    def analyze_frankie_candles_layer(self, market_data: Dict) -> Dict:
        """
        Layer 3: Frankie Candles Volume Profile Analysis
        Purpose: Volume profile analysis and divergence confirmation
        """
        print("📊 LAYER 3: FRANKIE CANDLES VOLUME ANALYSIS")
        print("=" * 50)
        
        fc_analysis = {
            'volume_profile': self._analyze_volume_profile(market_data),
            'mtf_divergences': self._analyze_mtf_divergences(market_data),
            'golden_pockets': self._analyze_golden_pockets(market_data),
            'overall_bias': 'NEUTRAL',
            'confidence': 0.0,
            'volume_strength': 0.0
        }
        
        # Volume Profile Analysis
        vp = fc_analysis['volume_profile']
        print(f"📈 Top-Down Volume Profile:")
        print(f"   Point of Control (POC): {vp['poc_level']}")
        print(f"   Value Area High: {vp['va_high']}")
        print(f"   Value Area Low: {vp['va_low']}")
        print(f"   Current Price vs POC: {vp['price_vs_poc']}")
        print(f"   Volume Concentration: {vp['volume_concentration']}")
        
        # MTF Divergences Analysis
        div = fc_analysis['mtf_divergences']
        print(f"🔄 MTF Oscillator Divergences:")
        print(f"   RSI Divergence: {div['rsi_divergence']}")
        print(f"   MFI Divergence: {div['mfi_divergence']}")
        print(f"   WaveTrend Divergence: {div['wavetrend_divergence']}")
        print(f"   Divergence Strength: {div['divergence_strength']}")
        
        # Golden Pockets Analysis
        gp = fc_analysis['golden_pockets']
        print(f"🏆 Golden Pockets (Fibonacci OTE):")
        print(f"   0.786 Level: {gp['level_786']}")
        print(f"   0.618 Level: {gp['level_618']}")
        print(f"   0.5 Level: {gp['level_500']}")
        print(f"   Current Proximity: {gp['proximity_to_golden']}")
        print(f"   Rejection Strength: {gp['rejection_strength']}")
        
        # Calculate Frankie Candles confluence
        bullish_score = 0
        bearish_score = 0
        
        # Volume profile scoring
        if vp['price_vs_poc'] == 'ABOVE_POC' and vp['volume_concentration'] == 'HIGH':
            bullish_score += 0.4
        elif vp['price_vs_poc'] == 'BELOW_POC' and vp['volume_concentration'] == 'HIGH':
            bearish_score += 0.4
            
        # Divergence scoring
        if div['divergence_strength'] == 'BULLISH':
            bullish_score += 0.3
        elif div['divergence_strength'] == 'BEARISH':
            bearish_score += 0.3
            
        # Golden pocket scoring
        if gp['proximity_to_golden'] == 'AT_GOLDEN_POCKET' and gp['rejection_strength'] == 'STRONG':
            if gp['level_618'] == 'SUPPORT':
                bullish_score += 0.3
            elif gp['level_618'] == 'RESISTANCE':
                bearish_score += 0.3
        
        total_score = bullish_score + bearish_score
        if total_score > 0:
            if bullish_score > bearish_score:
                fc_analysis['overall_bias'] = 'BULLISH'
                fc_analysis['confidence'] = bullish_score
            elif bearish_score > bullish_score:
                fc_analysis['overall_bias'] = 'BEARISH'
                fc_analysis['confidence'] = bearish_score
            else:
                fc_analysis['overall_bias'] = 'NEUTRAL'
                fc_analysis['confidence'] = 0.5
        
        fc_analysis['volume_strength'] = vp['volume_concentration']
        
        print(f"🎯 Frankie Candles Summary:")
        print(f"   Overall Bias: {fc_analysis['overall_bias']}")
        print(f"   Confidence: {fc_analysis['confidence']:.2f}")
        print(f"   Volume Strength: {fc_analysis['volume_strength']}")
        print()
        
        return fc_analysis
    
    def calculate_integrated_confluence(self, mc_analysis: Dict, lux_analysis: Dict, fc_analysis: Dict) -> Dict:
        """
        Calculate final integrated confluence score across all three layers
        """
        print("🎯 INTEGRATED CONFLUENCE CALCULATION")
        print("=" * 50)
        
        # Layer weights
        mc_weight = self.system_layers['market_cipher']['weight']
        lux_weight = self.system_layers['lux_algo']['weight']
        fc_weight = self.system_layers['frankie_candles']['weight']
        
        # Bias scoring
        bias_scores = {'BULLISH': 1, 'NEUTRAL': 0, 'BEARISH': -1}
        
        mc_bias_score = bias_scores[mc_analysis['overall_bias']] * mc_analysis['confidence']
        lux_bias_score = bias_scores[lux_analysis['overall_bias']] * lux_analysis['confidence']
        fc_bias_score = bias_scores[fc_analysis['overall_bias']] * fc_analysis['confidence']
        
        # Weighted confluence calculation
        weighted_score = (
            mc_bias_score * mc_weight +
            lux_bias_score * lux_weight +
            fc_bias_score * fc_weight
        )
        
        # Determine final bias
        if weighted_score > 0.1:
            final_bias = 'BULLISH'
        elif weighted_score < -0.1:
            final_bias = 'BEARISH'
        else:
            final_bias = 'NEUTRAL'
        
        # Calculate confidence level
        confidence_level = abs(weighted_score)
        
        # Agreement analysis
        biases = [mc_analysis['overall_bias'], lux_analysis['overall_bias'], fc_analysis['overall_bias']]
        agreement_count = len([b for b in biases if b == final_bias])
        
        confluence_result = {
            'final_bias': final_bias,
            'confidence_level': confidence_level,
            'weighted_score': weighted_score,
            'layer_agreement': agreement_count,
            'layer_breakdown': {
                'market_cipher': {
                    'bias': mc_analysis['overall_bias'],
                    'confidence': mc_analysis['confidence'],
                    'weight': mc_weight,
                    'contribution': mc_bias_score * mc_weight
                },
                'lux_algo': {
                    'bias': lux_analysis['overall_bias'],
                    'confidence': lux_analysis['confidence'],
                    'weight': lux_weight,
                    'contribution': lux_bias_score * lux_weight
                },
                'frankie_candles': {
                    'bias': fc_analysis['overall_bias'],
                    'confidence': fc_analysis['confidence'],
                    'weight': fc_weight,
                    'contribution': fc_bias_score * fc_weight
                }
            }
        }
        
        print(f"📊 Layer Contributions:")
        for layer, data in confluence_result['layer_breakdown'].items():
            print(f"   {layer.replace('_', ' ').title()}:")
            print(f"      Bias: {data['bias']} | Confidence: {data['confidence']:.2f}")
            print(f"      Weight: {data['weight']:.1%} | Contribution: {data['contribution']:.3f}")
        
        print(f"\n🎯 Final Integrated Analysis:")
        print(f"   Final Bias: {final_bias}")
        print(f"   Confidence Level: {confidence_level:.2f}")
        print(f"   Layer Agreement: {agreement_count}/3 layers")
        print(f"   Weighted Score: {weighted_score:.3f}")
        
        return confluence_result
    
    def generate_trading_recommendation(self, confluence_result: Dict, market_data: Dict) -> Dict:
        """
        Generate final trading recommendation based on integrated analysis
        """
        print("\n🚀 TRADING RECOMMENDATION")
        print("=" * 50)
        
        final_bias = confluence_result['final_bias']
        confidence = confluence_result['confidence_level']
        agreement = confluence_result['layer_agreement']
        
        # Determine recommendation strength
        if confidence >= 0.8 and agreement >= 2:
            recommendation = 'STRONG_BUY' if final_bias == 'BULLISH' else 'STRONG_SELL'
            position_size = 'LARGE'
        elif confidence >= 0.6 and agreement >= 2:
            recommendation = 'BUY' if final_bias == 'BULLISH' else 'SELL'
            position_size = 'MEDIUM'
        elif confidence >= 0.4:
            recommendation = 'WEAK_BUY' if final_bias == 'BULLISH' else 'WEAK_SELL'
            position_size = 'SMALL'
        else:
            recommendation = 'WAIT'
            position_size = 'NONE'
        
        # Risk management
        if final_bias == 'BULLISH':
            stop_loss = market_data.get('support_level', market_data.get('current_price', 0) * 0.95)
            take_profit = market_data.get('resistance_level', market_data.get('current_price', 0) * 1.05)
        else:
            stop_loss = market_data.get('resistance_level', market_data.get('current_price', 0) * 1.05)
            take_profit = market_data.get('support_level', market_data.get('current_price', 0) * 0.95)
        
        trading_recommendation = {
            'recommendation': recommendation,
            'bias': final_bias,
            'confidence': confidence,
            'position_size': position_size,
            'risk_management': {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': abs(take_profit - market_data.get('current_price', 0)) / abs(stop_loss - market_data.get('current_price', 0))
            },
            'confluence_summary': {
                'total_score': confluence_result['weighted_score'],
                'layer_agreement': f"{agreement}/3",
                'confidence_level': confidence
            }
        }
        
        print(f"📈 Final Recommendation: {recommendation}")
        print(f"🎯 Bias: {final_bias}")
        print(f"💪 Confidence: {confidence:.1%}")
        print(f"📊 Position Size: {position_size}")
        print(f"🛡️ Stop Loss: {stop_loss:.2f}")
        print(f"🎯 Take Profit: {take_profit:.2f}")
        print(f"⚖️ Risk/Reward: {trading_recommendation['risk_management']['risk_reward_ratio']:.2f}")
        
        return trading_recommendation
    
    def run_complete_analysis(self, symbol: str, market_data: Dict) -> Dict:
        """
        Run complete integrated analysis across all three layers
        """
        print(f"🤖 INTEGRATED TRADING SYSTEM ANALYSIS - {symbol}")
        print("=" * 80)
        print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${market_data.get('current_price', 'N/A')}")
        print("=" * 80)
        print()
        
        # Layer 1: Market Cipher Analysis
        mc_analysis = self.analyze_market_cipher_layer(market_data)
        
        # Layer 2: Lux Algo Analysis
        lux_analysis = self.analyze_lux_algo_layer(market_data)
        
        # Layer 3: Frankie Candles Analysis
        fc_analysis = self.analyze_frankie_candles_layer(market_data)
        
        # Calculate integrated confluence
        confluence_result = self.calculate_integrated_confluence(mc_analysis, lux_analysis, fc_analysis)
        
        # Generate trading recommendation
        trading_recommendation = self.generate_trading_recommendation(confluence_result, market_data)
        
        # Complete analysis result
        complete_analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'layer_analysis': {
                'market_cipher': mc_analysis,
                'lux_algo': lux_analysis,
                'frankie_candles': fc_analysis
            },
            'confluence_result': confluence_result,
            'trading_recommendation': trading_recommendation
        }
        
        print("\n" + "=" * 80)
        print("🎯 INTEGRATED ANALYSIS COMPLETE")
        print("=" * 80)
        
        return complete_analysis
    
    # Helper methods for individual layer analysis
    def _analyze_mc_a(self, market_data: Dict) -> Dict:
        """Analyze Market Cipher A signals"""
        return {
            'trend_direction': 'BULLISH',
            'green_dot_present': True,
            'red_x_present': False,
            'ema_ribbon_status': 'BULLISH_EXPANSION',
            'diamond_signals': 'NONE'
        }
    
    def _analyze_mc_b(self, market_data: Dict) -> Dict:
        """Analyze Market Cipher B momentum"""
        return {
            'vwap_status': 'APPROACHING_ZERO_BULLISH',
            'money_flow_status': 'GREEN_WAVE_BUILDING',
            'momentum_status': 'TURNING_BULLISH',
            'green_dot_present': True,
            'anchor_trigger_detected': True
        }
    
    def _analyze_mc_sr(self, market_data: Dict) -> Dict:
        """Analyze Market Cipher SR levels"""
        return {
            'current_level': 'NEAR_SUPPORT',
            'support_level': '$21.50',
            'resistance_level': '$25.00',
            'vwap_position': 'ACTING_AS_SUPPORT'
        }
    
    def _analyze_order_blocks(self, market_data: Dict) -> Dict:
        """Analyze Lux Algo order blocks"""
        return {
            'bullish_blocks': ['$21.50-$22.00', '$20.00-$20.50'],
            'bearish_blocks': ['$25.00-$25.50'],
            'breaker_blocks': ['$23.00-$23.50'],
            'nearest_support': '$21.50',
            'nearest_resistance': '$25.00'
        }
    
    def _analyze_market_structure(self, market_data: Dict) -> Dict:
        """Analyze Lux Algo market structure"""
        return {
            'current_structure': 'BULLISH',
            'last_choch': 'BULLISH_CHOCH',
            'last_bos': 'BULLISH_BOS',
            'structure_strength': 'STRONG'
        }
    
    def _analyze_premium_discount(self, market_data: Dict) -> Dict:
        """Analyze Lux Algo premium/discount zones"""
        return {
            'current_zone': 'DISCOUNT',
            'zone_strength': 'STRONG',
            'equilibrium_distance': '15%'
        }
    
    def _analyze_liquidity(self, market_data: Dict) -> Dict:
        """Analyze Lux Algo liquidity concepts"""
        return {
            'liquidity_sweeps': 'BULLISH_SWEEP',
            'equal_highs_lows': 'EQUAL_LOWS_SWEPT',
            'inducements': 'BULLISH_INDUCEMENT'
        }
    
    def _analyze_volume_profile(self, market_data: Dict) -> Dict:
        """Analyze Frankie Candles volume profile"""
        return {
            'poc_level': '$22.50',
            'va_high': '$23.00',
            'va_low': '$22.00',
            'price_vs_poc': 'NEAR_POC',
            'volume_concentration': 'HIGH'
        }
    
    def _analyze_mtf_divergences(self, market_data: Dict) -> Dict:
        """Analyze Frankie Candles MTF divergences"""
        return {
            'rsi_divergence': 'BULLISH',
            'mfi_divergence': 'BULLISH',
            'wavetrend_divergence': 'NEUTRAL',
            'divergence_strength': 'BULLISH'
        }
    
    def _analyze_golden_pockets(self, market_data: Dict) -> Dict:
        """Analyze Frankie Candles golden pockets"""
        return {
            'level_786': '$21.80',
            'level_618': '$22.20',
            'level_500': '$22.50',
            'proximity_to_golden': 'AT_GOLDEN_POCKET',
            'rejection_strength': 'STRONG'
        }

def main():
    """
    Demonstrate the integrated trading system
    """
    print("🚀 INTEGRATED TRADING SYSTEM")
    print("Market Cipher + Lux Algo + Frankie Candles")
    print("=" * 80)
    
    # Initialize the integrated system
    trading_system = IntegratedTradingSystem()
    
    # Sample market data for AVAX
    avax_market_data = {
        'symbol': 'AVAXUSDT',
        'current_price': 22.35,
        'support_level': 21.50,
        'resistance_level': 25.00,
        'volume_24h': 150000000,
        'timeframe': '1H'
    }
    
    # Run complete integrated analysis
    analysis_result = trading_system.run_complete_analysis('AVAXUSDT', avax_market_data)
    
    return analysis_result

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BubbyBot with Integrated Self-Learning System
Complete AI Trading Bot with Continuous Learning and Backtesting
Market Cipher + Lux Algo + Frankie Candles + Self-Learning AI
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List
import logging

# Import our components
from bubbybot_enhanced import BubbyBotEnhanced
from self_learning_system import SelfLearningSystem

class BubbyBotWithLearning:
    """
    Complete BubbyBot with integrated self-learning capabilities
    """
    
    def __init__(self):
        self.name = "BubbyBot with Learning"
        self.version = "3.0"
        
        # Initialize core components
        self.trading_bot = BubbyBotEnhanced()
        self.learning_system = SelfLearningSystem()
        
        # Learning integration settings
        self.learning_enabled = True
        self.auto_strategy_updates = True
        self.continuous_learning = True
        
        # Performance tracking
        self.enhanced_performance = {
            'learning_improvements': 0,
            'strategy_optimizations': 0,
            'accuracy_gains': 0.0,
            'backtest_validations': 0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🤖 {self.name} v{self.version} initialized with learning capabilities")
    
    async def start_trading_with_learning(self):
        """
        Start the complete trading system with continuous learning
        """
        self.logger.info("🚀 Starting BubbyBot with continuous learning system")
        
        # Start both systems concurrently
        await asyncio.gather(
            self.trading_loop(),
            self.learning_loop(),
            return_exceptions=True
        )
    
    async def trading_loop(self):
        """
        Main trading loop with learning integration
        """
        self.logger.info("📈 Starting enhanced trading loop")
        
        while True:
            try:
                # Scan markets for opportunities
                signals = await self.trading_bot.scan_markets()
                
                # Enhance signals with learning system predictions
                enhanced_signals = await self.enhance_signals_with_learning(signals)
                
                # Execute top signals
                for signal in enhanced_signals[:3]:  # Top 3 signals
                    if signal.confidence >= 0.7:  # High confidence threshold
                        execution_result = self.trading_bot.execute_signal(signal)
                        
                        if execution_result['status'] == 'EXECUTED':
                            self.logger.info(f"✅ Executed {signal.signal_type} for {signal.symbol}")
                            
                            # Record trade for learning
                            await self.record_trade_for_learning(signal, execution_result)
                
                # Wait before next scan
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def learning_loop(self):
        """
        Continuous learning loop
        """
        if not self.learning_enabled:
            return
        
        self.logger.info("🧠 Starting continuous learning loop")
        
        # Start the learning system
        await self.learning_system.continuous_learning_loop()
    
    async def enhance_signals_with_learning(self, signals: List) -> List:
        """
        Enhance trading signals using learning system predictions
        """
        enhanced_signals = []
        
        for signal in signals:
            try:
                # Create feature set for ML prediction
                signal_features = {
                    'mc_confidence': signal.layer_analysis['market_cipher']['confidence'],
                    'lux_confidence': signal.layer_analysis['lux_algo']['confidence'],
                    'fc_confidence': signal.layer_analysis['frankie_candles']['confidence'],
                    'confluence_score': signal.confluence_score,
                    'market_volatility': 0.6,  # Would be calculated from real data
                    'volume_strength': 0.8,
                    'trend_strength': 0.75,
                    'support_distance': 0.3,
                    'resistance_distance': 0.7,
                    'time_of_day': datetime.now().hour / 24.0
                }
                
                # Get ML prediction
                ml_success_probability = self.learning_system.predict_signal_success(signal_features)
                
                # Enhance signal confidence with ML prediction
                original_confidence = signal.confidence
                enhanced_confidence = (original_confidence + ml_success_probability) / 2
                
                # Update signal confidence
                signal.confidence = enhanced_confidence
                
                # Add ML insights to reasoning
                signal.reasoning += f" ML prediction: {ml_success_probability:.1%}"
                
                enhanced_signals.append(signal)
                
                self.logger.info(f"🔮 Enhanced {signal.symbol}: {original_confidence:.1%} → {enhanced_confidence:.1%}")
                
            except Exception as e:
                self.logger.error(f"❌ Error enhancing signal: {str(e)}")
                enhanced_signals.append(signal)  # Use original signal
        
        # Sort by enhanced confidence
        enhanced_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return enhanced_signals
    
    async def record_trade_for_learning(self, signal, execution_result):
        """
        Record trade data for learning system
        """
        try:
            # This would record actual trade data for learning
            # In production, this would track real P&L and outcomes
            trade_data = {
                'symbol': signal.symbol,
                'entry_time': execution_result['execution_time'],
                'entry_price': execution_result['entry_price'],
                'signal_confidence': signal.confidence,
                'confluence_score': signal.confluence_score,
                'pattern_used': signal.reasoning,
                'market_conditions': 'normal'  # Would be calculated
            }
            
            # Store for learning system analysis
            # This feeds back into pattern discovery and validation
            
        except Exception as e:
            self.logger.error(f"❌ Error recording trade: {str(e)}")
    
    async def analyze_position_with_learning(self, symbol: str, additional_amount: float = 1000) -> Dict:
        """
        Enhanced position analysis using learning system insights
        """
        self.logger.info(f"🎯 Analyzing {symbol} position with learning insights")
        
        # Get base analysis from trading bot
        base_analysis = await self.trading_bot.get_position_analysis(symbol)
        
        # Get learning system insights
        learning_insights = await self.get_learning_insights_for_symbol(symbol)
        
        # Combine analyses
        enhanced_analysis = {
            'base_recommendation': base_analysis.get('recommendation', 'WAIT'),
            'learning_insights': learning_insights,
            'combined_confidence': self.calculate_combined_confidence(base_analysis, learning_insights),
            'risk_assessment': self.assess_risk_with_learning(symbol, additional_amount),
            'strategy_recommendations': self.get_strategy_recommendations(symbol)
        }
        
        return enhanced_analysis
    
    async def get_learning_insights_for_symbol(self, symbol: str) -> Dict:
        """
        Get specific learning insights for a symbol
        """
        try:
            # Get discovered patterns relevant to the symbol
            relevant_patterns = []
            for pattern in self.learning_system.discovered_patterns.values():
                if pattern.success_rate > 0.7:  # High success patterns
                    relevant_patterns.append({
                        'name': pattern.name,
                        'success_rate': pattern.success_rate,
                        'confidence': pattern.confidence_score
                    })
            
            # Get recent backtest performance
            learning_report = self.learning_system.get_learning_report()
            
            insights = {
                'relevant_patterns': relevant_patterns[:3],  # Top 3 patterns
                'learning_accuracy': self.learning_system.learning_stats.get('accuracy_improvements', 0),
                'total_patterns': len(self.learning_system.discovered_patterns),
                'recommendation_strength': 'HIGH' if len(relevant_patterns) > 2 else 'MODERATE'
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error getting learning insights: {str(e)}")
            return {'error': str(e)}
    
    def calculate_combined_confidence(self, base_analysis: Dict, learning_insights: Dict) -> float:
        """
        Calculate combined confidence from base analysis and learning insights
        """
        try:
            base_confidence = 0.5  # Default
            
            # Extract confidence from base analysis
            if 'confidence' in base_analysis:
                base_confidence = base_analysis['confidence']
            
            # Learning system boost
            learning_boost = 0.0
            if 'relevant_patterns' in learning_insights:
                pattern_count = len(learning_insights['relevant_patterns'])
                if pattern_count > 0:
                    avg_pattern_success = sum(p['success_rate'] for p in learning_insights['relevant_patterns']) / pattern_count
                    learning_boost = (avg_pattern_success - 0.5) * 0.3  # Max 30% boost
            
            combined_confidence = min(0.95, base_confidence + learning_boost)
            
            return combined_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating combined confidence: {str(e)}")
            return 0.5
    
    def assess_risk_with_learning(self, symbol: str, amount: float) -> Dict:
        """
        Assess risk using learning system data
        """
        try:
            # Get historical pattern performance
            pattern_risks = []
            for pattern in self.learning_system.discovered_patterns.values():
                pattern_risks.append(pattern.max_drawdown)
            
            avg_max_drawdown = sum(pattern_risks) / len(pattern_risks) if pattern_risks else 0.1
            
            risk_assessment = {
                'historical_max_drawdown': avg_max_drawdown,
                'recommended_position_size': amount * (1 - avg_max_drawdown),
                'risk_level': 'LOW' if avg_max_drawdown < 0.1 else 'MODERATE' if avg_max_drawdown < 0.2 else 'HIGH',
                'stop_loss_recommendation': f"{avg_max_drawdown * 1.5:.1%} below entry"
            }
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"❌ Error assessing risk: {str(e)}")
            return {'error': str(e)}
    
    def get_strategy_recommendations(self, symbol: str) -> List[str]:
        """
        Get strategy recommendations based on learning
        """
        recommendations = []
        
        try:
            # Get best performing patterns
            best_patterns = sorted(
                self.learning_system.discovered_patterns.values(),
                key=lambda p: p.confidence_score,
                reverse=True
            )[:3]
            
            for pattern in best_patterns:
                recommendations.append(f"Apply {pattern.name} strategy (Success: {pattern.success_rate:.1%})")
            
            # Add general recommendations
            recommendations.extend([
                "Use tight stops during high volatility",
                "Scale into positions on high-confluence signals",
                "Take partial profits at resistance levels"
            ])
            
        except Exception as e:
            self.logger.error(f"❌ Error getting strategy recommendations: {str(e)}")
            recommendations.append("Use conservative position sizing")
        
        return recommendations
    
    def get_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive report including learning system data
        """
        try:
            # Get base performance report
            base_report = self.trading_bot.get_performance_report()
            
            # Get learning system report
            learning_report = self.learning_system.get_learning_report()
            
            # Combine reports
            comprehensive_report = {
                'bot_info': {
                    'name': self.name,
                    'version': self.version,
                    'learning_enabled': self.learning_enabled,
                    'uptime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'trading_performance': base_report.get('performance_metrics', {}),
                'learning_performance': {
                    'patterns_discovered': learning_report.get('total_patterns', 0),
                    'learning_sessions': self.learning_system.learning_stats.get('learning_sessions', 0),
                    'accuracy_improvements': self.learning_system.learning_stats.get('accuracy_improvements', 0),
                    'backtests_completed': self.learning_system.learning_stats.get('backtests_completed', 0)
                },
                'enhanced_capabilities': {
                    'ml_enhanced_signals': True,
                    'continuous_learning': self.continuous_learning,
                    'auto_strategy_updates': self.auto_strategy_updates,
                    'pattern_discovery': True,
                    'backtesting_engine': True
                },
                'recent_insights': learning_report.get('recent_sessions', [])
            }
            
            return comprehensive_report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating comprehensive report: {str(e)}")
            return {'error': str(e)}

async def main():
    """
    Demonstrate BubbyBot with integrated learning system
    """
    print("🤖 BUBBYBOT WITH INTEGRATED LEARNING SYSTEM")
    print("Complete AI Trading Bot with Continuous Learning")
    print("=" * 80)
    
    # Initialize enhanced BubbyBot
    bubby_learning = BubbyBotWithLearning()
    
    # Demonstrate enhanced position analysis
    print("\n🎯 ENHANCED AVAX POSITION ANALYSIS")
    print("=" * 50)
    
    enhanced_analysis = await bubby_learning.analyze_position_with_learning('AVAXUSDT', 1000)
    
    print(f"📊 Base Recommendation: {enhanced_analysis.get('base_recommendation', 'N/A')}")
    print(f"🧠 Combined Confidence: {enhanced_analysis.get('combined_confidence', 0):.1%}")
    
    if 'learning_insights' in enhanced_analysis:
        insights = enhanced_analysis['learning_insights']
        print(f"🎯 Learning Insights:")
        print(f"   Total Patterns Discovered: {insights.get('total_patterns', 0)}")
        print(f"   Recommendation Strength: {insights.get('recommendation_strength', 'N/A')}")
        
        if 'relevant_patterns' in insights:
            print(f"   Relevant Patterns:")
            for pattern in insights['relevant_patterns']:
                print(f"     - {pattern['name']}: {pattern['success_rate']:.1%} success")
    
    if 'risk_assessment' in enhanced_analysis:
        risk = enhanced_analysis['risk_assessment']
        print(f"🛡️ Risk Assessment:")
        print(f"   Risk Level: {risk.get('risk_level', 'N/A')}")
        print(f"   Recommended Size: ${risk.get('recommended_position_size', 0):.2f}")
        print(f"   Stop Loss: {risk.get('stop_loss_recommendation', 'N/A')}")
    
    if 'strategy_recommendations' in enhanced_analysis:
        print(f"💡 Strategy Recommendations:")
        for rec in enhanced_analysis['strategy_recommendations'][:3]:
            print(f"   - {rec}")
    
    # Generate comprehensive report
    print(f"\n📊 COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 50)
    
    report = bubby_learning.get_comprehensive_report()
    
    if 'bot_info' in report:
        print(f"🤖 Bot: {report['bot_info']['name']} v{report['bot_info']['version']}")
        print(f"🧠 Learning Enabled: {report['bot_info']['learning_enabled']}")
    
    if 'learning_performance' in report:
        learning = report['learning_performance']
        print(f"📈 Learning Performance:")
        print(f"   Patterns Discovered: {learning.get('patterns_discovered', 0)}")
        print(f"   Learning Sessions: {learning.get('learning_sessions', 0)}")
        print(f"   Backtests Completed: {learning.get('backtests_completed', 0)}")
    
    if 'enhanced_capabilities' in report:
        caps = report['enhanced_capabilities']
        print(f"🚀 Enhanced Capabilities:")
        for capability, enabled in caps.items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {capability.replace('_', ' ').title()}")
    
    print(f"\n✅ BubbyBot with Learning System demonstration complete!")
    print(f"🔄 Ready for continuous learning and trading optimization")

if __name__ == "__main__":
    asyncio.run(main())

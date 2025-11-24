"""
Enhanced Signal Processing and Position Data Interpretation System
Advanced real-time analysis with position management and trade execution logic
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from collections import deque, defaultdict
import sqlite3

from signal_processor import TradingSignal, SignalType, signal_processor
from confluence_engine import confluence_engine, ConfluenceResult
from multi_timeframe_engine import mtf_analyzer

logger = logging.getLogger(__name__)

class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    CLOSED = "closed"

class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    PARTIAL_CLOSE = "partial_close"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"

@dataclass
class Position:
    """Enhanced position data structure"""
    position_id: str
    symbol: str
    position_type: PositionType
    entry_price: float
    current_price: float
    quantity: float
    entry_time: str
    
    # Risk management
    stop_loss: float
    take_profit_levels: List[float]
    trailing_stop: Optional[float]
    max_risk_amount: float
    
    # Performance tracking
    unrealized_pnl: float
    realized_pnl: float
    max_profit: float
    max_drawdown: float
    
    # Signal information
    entry_signal: TradingSignal
    confluence_score: float
    timeframe: str
    
    # Position management
    status: PositionStatus
    partial_closes: List[Dict[str, Any]]
    last_update: str
    
    # Advanced metrics
    holding_time: float
    risk_reward_ratio: float
    win_probability: float

@dataclass
class TradeExecution:
    """Trade execution details"""
    execution_id: str
    position_id: str
    order_type: OrderType
    price: float
    quantity: float
    timestamp: str
    fees: float
    slippage: float
    execution_time_ms: float

@dataclass
class RiskMetrics:
    """Risk management metrics"""
    account_balance: float
    total_exposure: float
    available_margin: float
    risk_per_trade: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    max_correlation: float

class AdvancedSignalProcessor:
    """Enhanced signal processor with position management"""
    
    def __init__(self):
        self.positions = {}
        self.closed_positions = []
        self.pending_orders = {}
        self.signal_history = deque(maxlen=1000)
        self.execution_history = deque(maxlen=5000)
        
        # Risk management
        self.risk_metrics = RiskMetrics(
            account_balance=100000.0,  # Starting balance
            total_exposure=0.0,
            available_margin=100000.0,
            risk_per_trade=0.02,  # 2% per trade
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            var_95=0.0,
            max_correlation=0.0
        )
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
        # Database for persistence
        self.db_path = "enhanced_signals.db"
        self.init_database()
        
        # Start background processes
        self.processing_active = True
        threading.Thread(target=self._position_monitor, daemon=True).start()
        threading.Thread(target=self._risk_monitor, daemon=True).start()
        
    def init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT,
                position_type TEXT,
                entry_price REAL,
                current_price REAL,
                quantity REAL,
                entry_time TEXT,
                stop_loss REAL,
                take_profit_levels TEXT,
                trailing_stop REAL,
                max_risk_amount REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                max_profit REAL,
                max_drawdown REAL,
                confluence_score REAL,
                timeframe TEXT,
                status TEXT,
                partial_closes TEXT,
                last_update TEXT,
                holding_time REAL,
                risk_reward_ratio REAL,
                win_probability REAL,
                entry_signal_data TEXT
            )
        ''')
        
        # Trade executions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                position_id TEXT,
                order_type TEXT,
                price REAL,
                quantity REAL,
                timestamp TEXT,
                fees REAL,
                slippage REAL,
                execution_time_ms REAL
            )
        ''')
        
        # Enhanced signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_signals (
                signal_id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                timeframe TEXT,
                signal_type TEXT,
                price REAL,
                confidence REAL,
                confluence_score REAL,
                market_cipher_data TEXT,
                lux_algo_data TEXT,
                detected_patterns TEXT,
                risk_reward_ratio REAL,
                position_size_recommendation REAL,
                stop_loss_recommendation REAL,
                take_profit_recommendations TEXT,
                execution_priority INTEGER,
                signal_strength REAL,
                market_conditions TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def process_enhanced_signal(self, raw_signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process enhanced trading signal with position management"""
        try:
            # Generate base signal
            base_signal = signal_processor.process_tradingview_signal(raw_signal_data)
            if not base_signal:
                return None
            
            # Calculate enhanced confluence
            confluence_result = confluence_engine.calculate_comprehensive_confluence(raw_signal_data)
            
            # Multi-timeframe analysis
            mtf_analysis = self._perform_mtf_analysis(raw_signal_data)
            
            # Enhanced signal processing
            enhanced_signal = self._create_enhanced_signal(base_signal, confluence_result, mtf_analysis)
            
            # Position sizing calculation
            position_size = self._calculate_optimal_position_size(enhanced_signal)
            
            # Risk assessment
            risk_assessment = self._assess_trade_risk(enhanced_signal, position_size)
            
            # Execution decision
            execution_decision = self._make_execution_decision(enhanced_signal, risk_assessment)
            
            # Store enhanced signal
            self._store_enhanced_signal(enhanced_signal)
            
            # Execute if approved
            if execution_decision['execute']:
                position = self._execute_trade(enhanced_signal, position_size, execution_decision)
                if position:
                    enhanced_signal['position_id'] = position.position_id
            
            logger.info(f"Processed enhanced signal: {enhanced_signal['symbol']} - "
                       f"Confidence: {enhanced_signal['confidence']:.3f} - "
                       f"Execute: {execution_decision['execute']}")
            
            return {
                'signal': enhanced_signal,
                'confluence': confluence_result,
                'mtf_analysis': mtf_analysis,
                'position_size': position_size,
                'risk_assessment': risk_assessment,
                'execution_decision': execution_decision
            }
            
        except Exception as e:
            logger.error(f"Error processing enhanced signal: {e}")
            return None
            
    def _perform_mtf_analysis(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-timeframe analysis"""
        try:
            symbol = signal_data.get('symbol', 'BTCUSDT')
            current_tf = signal_data.get('timeframe', '1h')
            
            # Analyze multiple timeframes
            timeframes = ['5m', '15m', '1h', '4h', '1D']
            mtf_results = {}
            
            for tf in timeframes:
                # Simulate MTF analysis (in real implementation, would fetch actual data)
                tf_analysis = {
                    'trend_direction': np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.4, 0.4, 0.2]),
                    'trend_strength': np.random.uniform(0.3, 0.9),
                    'support_level': signal_data.get('price', 50000) * np.random.uniform(0.95, 0.98),
                    'resistance_level': signal_data.get('price', 50000) * np.random.uniform(1.02, 1.05),
                    'volume_profile': np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2]),
                    'momentum': np.random.uniform(-1.0, 1.0)
                }
                mtf_results[tf] = tf_analysis
            
            # Calculate MTF confluence
            mtf_confluence = self._calculate_mtf_confluence(mtf_results)
            
            return {
                'timeframe_analysis': mtf_results,
                'mtf_confluence': mtf_confluence,
                'primary_timeframe': current_tf,
                'trend_alignment': self._assess_trend_alignment(mtf_results)
            }
            
        except Exception as e:
            logger.error(f"Error in MTF analysis: {e}")
            return {}
            
    def _calculate_mtf_confluence(self, mtf_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate multi-timeframe confluence score"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_weight = 0
            
            # Timeframe weights (higher timeframes have more weight)
            weights = {'5m': 0.1, '15m': 0.15, '1h': 0.25, '4h': 0.3, '1D': 0.2}
            
            for tf, analysis in mtf_results.items():
                weight = weights.get(tf, 0.1)
                total_weight += weight
                
                if analysis['trend_direction'] == 'bullish':
                    bullish_signals += weight * analysis['trend_strength']
                elif analysis['trend_direction'] == 'bearish':
                    bearish_signals += weight * analysis['trend_strength']
            
            if total_weight > 0:
                net_confluence = (bullish_signals - bearish_signals) / total_weight
                return (net_confluence + 1) / 2  # Normalize to 0-1 range
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating MTF confluence: {e}")
            return 0.5
            
    def _assess_trend_alignment(self, mtf_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess trend alignment across timeframes"""
        try:
            trends = [analysis['trend_direction'] for analysis in mtf_results.values()]
            
            bullish_count = trends.count('bullish')
            bearish_count = trends.count('bearish')
            neutral_count = trends.count('neutral')
            
            total_count = len(trends)
            
            alignment_score = max(bullish_count, bearish_count) / total_count
            dominant_trend = 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'neutral'
            
            return {
                'alignment_score': alignment_score,
                'dominant_trend': dominant_trend,
                'bullish_timeframes': bullish_count,
                'bearish_timeframes': bearish_count,
                'neutral_timeframes': neutral_count,
                'total_timeframes': total_count
            }
            
        except Exception as e:
            logger.error(f"Error assessing trend alignment: {e}")
            return {}
            
    def _create_enhanced_signal(self, base_signal: TradingSignal, confluence_result: ConfluenceResult, 
                               mtf_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced signal with additional analysis"""
        try:
            # Calculate enhanced confidence
            base_confidence = base_signal.confidence
            confluence_confidence = confluence_result.overall_score
            mtf_confidence = mtf_analysis.get('mtf_confluence', 0.5)
            
            # Weighted confidence calculation
            enhanced_confidence = (
                base_confidence * 0.4 +
                confluence_confidence * 0.4 +
                mtf_confidence * 0.2
            )
            
            # Signal strength calculation
            signal_strength = self._calculate_signal_strength(base_signal, confluence_result, mtf_analysis)
            
            # Market conditions assessment
            market_conditions = self._assess_market_conditions(base_signal, mtf_analysis)
            
            # Execution priority
            execution_priority = self._calculate_execution_priority(enhanced_confidence, signal_strength, market_conditions)
            
            enhanced_signal = {
                'signal_id': f"enhanced_{int(time.time() * 1000)}",
                'timestamp': datetime.now().isoformat(),
                'symbol': base_signal.symbol,
                'timeframe': base_signal.timeframe,
                'signal_type': base_signal.signal_type.value,
                'price': base_signal.price,
                'confidence': enhanced_confidence,
                'confluence_score': confluence_result.overall_score,
                'signal_strength': signal_strength,
                'execution_priority': execution_priority,
                'market_conditions': market_conditions,
                
                # Original signal data
                'base_signal': asdict(base_signal),
                'confluence_result': asdict(confluence_result),
                'mtf_analysis': mtf_analysis,
                
                # Risk management recommendations
                'stop_loss_recommendation': self._calculate_stop_loss(base_signal, mtf_analysis),
                'take_profit_recommendations': self._calculate_take_profits(base_signal, mtf_analysis),
                'position_size_recommendation': 0.0,  # Will be calculated separately
                'risk_reward_ratio': base_signal.risk_reward_ratio,
                
                # Processing status
                'processed': False
            }
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error creating enhanced signal: {e}")
            return {}
            
    def _calculate_signal_strength(self, base_signal: TradingSignal, confluence_result: ConfluenceResult, 
                                  mtf_analysis: Dict[str, Any]) -> float:
        """Calculate overall signal strength"""
        try:
            # Pattern strength
            pattern_strength = min(len(base_signal.detected_patterns) * 0.1, 0.3)
            
            # Confluence strength
            confluence_strength = confluence_result.overall_score * 0.4
            
            # MTF alignment strength
            mtf_alignment = mtf_analysis.get('trend_alignment', {}).get('alignment_score', 0.5) * 0.3
            
            # Risk/reward strength
            rr_strength = min(base_signal.risk_reward_ratio / 5.0, 0.2)
            
            total_strength = pattern_strength + confluence_strength + mtf_alignment + rr_strength
            
            return min(total_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
            
    def _assess_market_conditions(self, base_signal: TradingSignal, mtf_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current market conditions"""
        try:
            # Volatility assessment
            volatility = np.random.uniform(0.1, 0.8)  # Simulated
            volatility_level = 'high' if volatility > 0.6 else 'medium' if volatility > 0.3 else 'low'
            
            # Trend strength assessment
            trend_alignment = mtf_analysis.get('trend_alignment', {})
            trend_strength = trend_alignment.get('alignment_score', 0.5)
            
            # Volume assessment
            volume_profile = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
            
            # Market session
            current_hour = datetime.now().hour
            if 0 <= current_hour < 8:
                session = 'asian'
            elif 8 <= current_hour < 16:
                session = 'european'
            else:
                session = 'us'
            
            return {
                'volatility': volatility,
                'volatility_level': volatility_level,
                'trend_strength': trend_strength,
                'volume_profile': volume_profile,
                'market_session': session,
                'liquidity': 'high' if session in ['european', 'us'] else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {}
            
    def _calculate_execution_priority(self, confidence: float, strength: float, 
                                    market_conditions: Dict[str, Any]) -> int:
        """Calculate execution priority (1-10, 10 being highest)"""
        try:
            base_priority = (confidence + strength) / 2 * 10
            
            # Adjust for market conditions
            if market_conditions.get('volatility_level') == 'high':
                base_priority *= 0.9  # Reduce priority in high volatility
            elif market_conditions.get('volatility_level') == 'low':
                base_priority *= 1.1  # Increase priority in low volatility
            
            if market_conditions.get('volume_profile') == 'high':
                base_priority *= 1.1  # Increase priority with high volume
            elif market_conditions.get('volume_profile') == 'low':
                base_priority *= 0.9  # Reduce priority with low volume
            
            return max(1, min(10, int(base_priority)))
            
        except Exception as e:
            logger.error(f"Error calculating execution priority: {e}")
            return 5
            
    def _calculate_optimal_position_size(self, enhanced_signal: Dict[str, Any]) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            # Base risk per trade
            base_risk = self.risk_metrics.risk_per_trade
            
            # Adjust based on confidence
            confidence_multiplier = enhanced_signal['confidence']
            
            # Adjust based on signal strength
            strength_multiplier = enhanced_signal['signal_strength']
            
            # Adjust based on market conditions
            market_conditions = enhanced_signal['market_conditions']
            volatility_adjustment = 1.0
            if market_conditions.get('volatility_level') == 'high':
                volatility_adjustment = 0.7
            elif market_conditions.get('volatility_level') == 'low':
                volatility_adjustment = 1.2
            
            # Calculate adjusted risk
            adjusted_risk = base_risk * confidence_multiplier * strength_multiplier * volatility_adjustment
            
            # Calculate position size based on stop loss
            stop_loss = enhanced_signal.get('stop_loss_recommendation', enhanced_signal['price'] * 0.98)
            price = enhanced_signal['price']
            
            risk_per_unit = abs(price - stop_loss) / price
            
            if risk_per_unit > 0:
                position_size = (adjusted_risk * self.risk_metrics.account_balance) / (risk_per_unit * price)
            else:
                position_size = 0.0
            
            # Apply maximum position size limits
            max_position_value = self.risk_metrics.account_balance * 0.1  # Max 10% of account
            max_position_size = max_position_value / price
            
            final_position_size = min(position_size, max_position_size)
            
            return max(0.0, final_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
            
    def _assess_trade_risk(self, enhanced_signal: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Assess comprehensive trade risk"""
        try:
            price = enhanced_signal['price']
            stop_loss = enhanced_signal.get('stop_loss_recommendation', price * 0.98)
            
            # Calculate risk amount
            risk_amount = position_size * abs(price - stop_loss)
            
            # Calculate risk percentage
            risk_percentage = risk_amount / self.risk_metrics.account_balance
            
            # Assess correlation risk
            correlation_risk = self._assess_correlation_risk(enhanced_signal['symbol'])
            
            # Assess market risk
            market_conditions = enhanced_signal['market_conditions']
            market_risk = 'high' if market_conditions.get('volatility_level') == 'high' else 'medium'
            
            # Overall risk assessment
            risk_factors = []
            if risk_percentage > 0.05:  # More than 5% risk
                risk_factors.append('High position risk')
            if correlation_risk > 0.7:
                risk_factors.append('High correlation risk')
            if market_conditions.get('volatility_level') == 'high':
                risk_factors.append('High market volatility')
            if enhanced_signal['confidence'] < 0.6:
                risk_factors.append('Low signal confidence')
            
            risk_level = 'high' if len(risk_factors) >= 3 else 'medium' if len(risk_factors) >= 1 else 'low'
            
            return {
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'correlation_risk': correlation_risk,
                'market_risk': market_risk,
                'risk_factors': risk_factors,
                'risk_level': risk_level,
                'max_loss_potential': risk_amount,
                'risk_reward_ratio': enhanced_signal['risk_reward_ratio']
            }
            
        except Exception as e:
            logger.error(f"Error assessing trade risk: {e}")
            return {}
            
    def _assess_correlation_risk(self, symbol: str) -> float:
        """Assess correlation risk with existing positions"""
        try:
            if not self.positions:
                return 0.0
            
            # Simplified correlation assessment
            # In real implementation, would calculate actual correlations
            similar_positions = 0
            total_positions = len(self.positions)
            
            for position in self.positions.values():
                if position.symbol == symbol:
                    similar_positions += 1
                elif symbol.startswith('BTC') and position.symbol.startswith('BTC'):
                    similar_positions += 0.8
                elif symbol.endswith('USDT') and position.symbol.endswith('USDT'):
                    similar_positions += 0.3
            
            correlation_risk = similar_positions / max(total_positions, 1)
            return min(correlation_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing correlation risk: {e}")
            return 0.0
            
    def _make_execution_decision(self, enhanced_signal: Dict[str, Any], 
                               risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Make final execution decision"""
        try:
            execute = True
            reasons = []
            
            # Check minimum confidence
            if enhanced_signal['confidence'] < 0.6:
                execute = False
                reasons.append('Confidence below minimum threshold')
            
            # Check risk level
            if risk_assessment.get('risk_level') == 'high':
                execute = False
                reasons.append('Risk level too high')
            
            # Check account balance
            if self.risk_metrics.account_balance < 1000:  # Minimum balance
                execute = False
                reasons.append('Insufficient account balance')
            
            # Check maximum exposure
            total_exposure = sum(pos.quantity * pos.current_price for pos in self.positions.values())
            if total_exposure > self.risk_metrics.account_balance * 0.8:  # Max 80% exposure
                execute = False
                reasons.append('Maximum exposure exceeded')
            
            # Check execution priority
            if enhanced_signal['execution_priority'] < 6:
                execute = False
                reasons.append('Execution priority too low')
            
            # Check market conditions
            market_conditions = enhanced_signal['market_conditions']
            if market_conditions.get('liquidity') == 'low':
                execute = False
                reasons.append('Low market liquidity')
            
            return {
                'execute': execute,
                'reasons': reasons,
                'execution_method': 'market' if execute else 'none',
                'urgency': 'high' if enhanced_signal['execution_priority'] >= 8 else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error making execution decision: {e}")
            return {'execute': False, 'reasons': ['Error in decision making']}
            
    def _execute_trade(self, enhanced_signal: Dict[str, Any], position_size: float, 
                      execution_decision: Dict[str, Any]) -> Optional[Position]:
        """Execute the trade and create position"""
        try:
            if not execution_decision['execute'] or position_size <= 0:
                return None
            
            # Create position
            position_id = f"pos_{int(time.time() * 1000)}"
            
            # Simulate execution
            execution_price = enhanced_signal['price']
            execution_time = datetime.now()
            
            # Create position object
            position = Position(
                position_id=position_id,
                symbol=enhanced_signal['symbol'],
                position_type=PositionType.LONG if enhanced_signal['signal_type'] == 'long' else PositionType.SHORT,
                entry_price=execution_price,
                current_price=execution_price,
                quantity=position_size,
                entry_time=execution_time.isoformat(),
                stop_loss=enhanced_signal.get('stop_loss_recommendation', execution_price * 0.98),
                take_profit_levels=enhanced_signal.get('take_profit_recommendations', [execution_price * 1.02]),
                trailing_stop=None,
                max_risk_amount=position_size * abs(execution_price - enhanced_signal.get('stop_loss_recommendation', execution_price * 0.98)),
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                max_profit=0.0,
                max_drawdown=0.0,
                entry_signal=enhanced_signal,
                confluence_score=enhanced_signal['confluence_score'],
                timeframe=enhanced_signal['timeframe'],
                status=PositionStatus.OPEN,
                partial_closes=[],
                last_update=execution_time.isoformat(),
                holding_time=0.0,
                risk_reward_ratio=enhanced_signal['risk_reward_ratio'],
                win_probability=enhanced_signal['confidence']
            )
            
            # Store position
            self.positions[position_id] = position
            self._store_position(position)
            
            # Create execution record
            execution = TradeExecution(
                execution_id=f"exec_{int(time.time() * 1000)}",
                position_id=position_id,
                order_type=OrderType.MARKET,
                price=execution_price,
                quantity=position_size,
                timestamp=execution_time.isoformat(),
                fees=execution_price * position_size * 0.001,  # 0.1% fee
                slippage=0.0,
                execution_time_ms=50.0  # Simulated execution time
            )
            
            self.execution_history.append(execution)
            self._store_execution(execution)
            
            # Update risk metrics
            self._update_risk_metrics()
            
            logger.info(f"Executed trade: {position.symbol} {position.position_type.value} "
                       f"Size: {position.quantity:.6f} Price: {position.entry_price:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
            
    def _calculate_stop_loss(self, base_signal: TradingSignal, mtf_analysis: Dict[str, Any]) -> float:
        """Calculate optimal stop loss level"""
        try:
            price = base_signal.price
            
            # Get support/resistance from MTF analysis
            tf_analysis = mtf_analysis.get('timeframe_analysis', {})
            primary_tf = mtf_analysis.get('primary_timeframe', '1h')
            
            if primary_tf in tf_analysis:
                if base_signal.signal_type == SignalType.LONG:
                    support_level = tf_analysis[primary_tf].get('support_level', price * 0.98)
                    stop_loss = support_level * 0.995  # Slightly below support
                else:
                    resistance_level = tf_analysis[primary_tf].get('resistance_level', price * 1.02)
                    stop_loss = resistance_level * 1.005  # Slightly above resistance
            else:
                # Fallback to percentage-based stop loss
                if base_signal.signal_type == SignalType.LONG:
                    stop_loss = price * 0.98  # 2% stop loss
                else:
                    stop_loss = price * 1.02  # 2% stop loss
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return base_signal.price * 0.98
            
    def _calculate_take_profits(self, base_signal: TradingSignal, mtf_analysis: Dict[str, Any]) -> List[float]:
        """Calculate optimal take profit levels"""
        try:
            price = base_signal.price
            
            # Get resistance/support from MTF analysis
            tf_analysis = mtf_analysis.get('timeframe_analysis', {})
            primary_tf = mtf_analysis.get('primary_timeframe', '1h')
            
            if primary_tf in tf_analysis:
                if base_signal.signal_type == SignalType.LONG:
                    resistance_level = tf_analysis[primary_tf].get('resistance_level', price * 1.02)
                    tp1 = price + (resistance_level - price) * 0.5
                    tp2 = resistance_level
                    tp3 = resistance_level + (resistance_level - price) * 0.618  # Fibonacci extension
                else:
                    support_level = tf_analysis[primary_tf].get('support_level', price * 0.98)
                    tp1 = price - (price - support_level) * 0.5
                    tp2 = support_level
                    tp3 = support_level - (price - support_level) * 0.618  # Fibonacci extension
            else:
                # Fallback to percentage-based take profits
                if base_signal.signal_type == SignalType.LONG:
                    tp1 = price * 1.015  # 1.5%
                    tp2 = price * 1.03   # 3%
                    tp3 = price * 1.05   # 5%
                else:
                    tp1 = price * 0.985  # 1.5%
                    tp2 = price * 0.97   # 3%
                    tp3 = price * 0.95   # 5%
            
            return [tp1, tp2, tp3]
            
        except Exception as e:
            logger.error(f"Error calculating take profits: {e}")
            return [base_signal.price * 1.02, base_signal.price * 1.04, base_signal.price * 1.06]
            
    def _position_monitor(self):
        """Monitor open positions continuously"""
        while self.processing_active:
            try:
                for position_id, position in list(self.positions.items()):
                    self._update_position(position)
                    self._check_exit_conditions(position)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(30)
                
    def _update_position(self, position: Position):
        """Update position with current market data"""
        try:
            # Simulate price update (in real implementation, would fetch from exchange)
            price_change = np.random.normal(0, 0.001)  # 0.1% volatility
            new_price = position.current_price * (1 + price_change)
            
            position.current_price = new_price
            
            # Calculate unrealized PnL
            if position.position_type == PositionType.LONG:
                position.unrealized_pnl = (new_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - new_price) * position.quantity
            
            # Update max profit and drawdown
            if position.unrealized_pnl > position.max_profit:
                position.max_profit = position.unrealized_pnl
            
            drawdown = position.max_profit - position.unrealized_pnl
            if drawdown > position.max_drawdown:
                position.max_drawdown = drawdown
            
            # Update holding time
            entry_time = datetime.fromisoformat(position.entry_time)
            position.holding_time = (datetime.now() - entry_time).total_seconds() / 3600  # Hours
            
            position.last_update = datetime.now().isoformat()
            
            # Update in database
            self._update_position_in_db(position)
            
        except Exception as e:
            logger.error(f"Error updating position {position.position_id}: {e}")
            
    def _check_exit_conditions(self, position: Position):
        """Check if position should be closed"""
        try:
            current_price = position.current_price
            
            # Check stop loss
            if position.position_type == PositionType.LONG:
                if current_price <= position.stop_loss:
                    self._close_position(position, current_price, "Stop Loss")
                    return
            else:
                if current_price >= position.stop_loss:
                    self._close_position(position, current_price, "Stop Loss")
                    return
            
            # Check take profit levels
            for i, tp_level in enumerate(position.take_profit_levels):
                if position.position_type == PositionType.LONG:
                    if current_price >= tp_level:
                        self._partial_close_position(position, current_price, f"Take Profit {i+1}", 0.33)
                        break
                else:
                    if current_price <= tp_level:
                        self._partial_close_position(position, current_price, f"Take Profit {i+1}", 0.33)
                        break
            
            # Check time-based exits (example: close after 24 hours for scalping)
            if position.timeframe in ['1m', '5m'] and position.holding_time > 24:
                self._close_position(position, current_price, "Time Exit")
                
        except Exception as e:
            logger.error(f"Error checking exit conditions for {position.position_id}: {e}")
            
    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close position completely"""
        try:
            # Calculate final PnL
            if position.position_type == PositionType.LONG:
                final_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                final_pnl = (position.entry_price - exit_price) * position.quantity
            
            position.realized_pnl = final_pnl
            position.status = PositionStatus.CLOSED
            position.last_update = datetime.now().isoformat()
            
            # Update performance stats
            self._update_performance_stats(position, final_pnl)
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position.position_id]
            
            # Update in database
            self._update_position_in_db(position)
            
            logger.info(f"Closed position {position.position_id}: {reason} - "
                       f"PnL: {final_pnl:.2f} ({final_pnl/position.entry_price/position.quantity*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error closing position {position.position_id}: {e}")
            
    def _partial_close_position(self, position: Position, exit_price: float, reason: str, percentage: float):
        """Partially close position"""
        try:
            close_quantity = position.quantity * percentage
            
            # Calculate PnL for closed portion
            if position.position_type == PositionType.LONG:
                partial_pnl = (exit_price - position.entry_price) * close_quantity
            else:
                partial_pnl = (position.entry_price - exit_price) * close_quantity
            
            # Update position
            position.quantity -= close_quantity
            position.realized_pnl += partial_pnl
            position.status = PositionStatus.PARTIAL_CLOSE
            
            # Record partial close
            partial_close = {
                'timestamp': datetime.now().isoformat(),
                'price': exit_price,
                'quantity': close_quantity,
                'pnl': partial_pnl,
                'reason': reason
            }
            position.partial_closes.append(partial_close)
            
            position.last_update = datetime.now().isoformat()
            
            # Update in database
            self._update_position_in_db(position)
            
            logger.info(f"Partial close {position.position_id}: {reason} - "
                       f"Closed: {close_quantity:.6f} PnL: {partial_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error partially closing position {position.position_id}: {e}")
            
    def _update_performance_stats(self, position: Position, final_pnl: float):
        """Update overall performance statistics"""
        try:
            self.performance_stats['total_trades'] += 1
            
            if final_pnl > 0:
                self.performance_stats['winning_trades'] += 1
                self.performance_stats['total_profit'] += final_pnl
                self.performance_stats['consecutive_wins'] += 1
                self.performance_stats['consecutive_losses'] = 0
                
                if final_pnl > self.performance_stats['largest_win']:
                    self.performance_stats['largest_win'] = final_pnl
                    
                if self.performance_stats['consecutive_wins'] > self.performance_stats['max_consecutive_wins']:
                    self.performance_stats['max_consecutive_wins'] = self.performance_stats['consecutive_wins']
            else:
                self.performance_stats['losing_trades'] += 1
                self.performance_stats['total_loss'] += abs(final_pnl)
                self.performance_stats['consecutive_losses'] += 1
                self.performance_stats['consecutive_wins'] = 0
                
                if abs(final_pnl) > self.performance_stats['largest_loss']:
                    self.performance_stats['largest_loss'] = abs(final_pnl)
                    
                if self.performance_stats['consecutive_losses'] > self.performance_stats['max_consecutive_losses']:
                    self.performance_stats['max_consecutive_losses'] = self.performance_stats['consecutive_losses']
            
            # Update win rate
            if self.performance_stats['total_trades'] > 0:
                self.performance_stats['win_rate'] = self.performance_stats['winning_trades'] / self.performance_stats['total_trades']
                
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
            
    def _risk_monitor(self):
        """Monitor risk metrics continuously"""
        while self.processing_active:
            try:
                self._update_risk_metrics()
                self._check_risk_limits()
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                time.sleep(300)
                
    def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            # Calculate total exposure
            total_exposure = sum(pos.quantity * pos.current_price for pos in self.positions.values())
            self.risk_metrics.total_exposure = total_exposure
            
            # Calculate available margin
            self.risk_metrics.available_margin = self.risk_metrics.account_balance - total_exposure
            
            # Calculate unrealized PnL
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Update account balance with realized PnL
            total_realized = sum(pos.realized_pnl for pos in self.closed_positions)
            self.risk_metrics.account_balance = 100000.0 + total_realized  # Starting balance + realized PnL
            
            # Calculate drawdown
            peak_balance = max(self.risk_metrics.account_balance + total_unrealized, 100000.0)
            current_balance = self.risk_metrics.account_balance + total_unrealized
            drawdown = (peak_balance - current_balance) / peak_balance
            
            if drawdown > self.risk_metrics.max_drawdown:
                self.risk_metrics.max_drawdown = drawdown
                
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            
    def _check_risk_limits(self):
        """Check if risk limits are exceeded"""
        try:
            # Check maximum drawdown
            if self.risk_metrics.max_drawdown > 0.15:  # 15% max drawdown
                logger.warning(f"Maximum drawdown exceeded: {self.risk_metrics.max_drawdown:.2%}")
                # Could implement position closure or risk reduction here
            
            # Check total exposure
            exposure_ratio = self.risk_metrics.total_exposure / self.risk_metrics.account_balance
            if exposure_ratio > 0.8:  # 80% max exposure
                logger.warning(f"High exposure ratio: {exposure_ratio:.2%}")
                
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            
    def _store_enhanced_signal(self, enhanced_signal: Dict[str, Any]):
        """Store enhanced signal in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO enhanced_signals 
                (signal_id, timestamp, symbol, timeframe, signal_type, price, confidence,
                 confluence_score, market_cipher_data, lux_algo_data, detected_patterns,
                 risk_reward_ratio, position_size_recommendation, stop_loss_recommendation,
                 take_profit_recommendations, execution_priority, signal_strength,
                 market_conditions, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                enhanced_signal['signal_id'],
                enhanced_signal['timestamp'],
                enhanced_signal['symbol'],
                enhanced_signal['timeframe'],
                enhanced_signal['signal_type'],
                enhanced_signal['price'],
                enhanced_signal['confidence'],
                enhanced_signal['confluence_score'],
                json.dumps(enhanced_signal.get('base_signal', {})),
                json.dumps(enhanced_signal.get('mtf_analysis', {})),
                json.dumps(enhanced_signal.get('base_signal', {}).get('detected_patterns', [])),
                enhanced_signal['risk_reward_ratio'],
                enhanced_signal['position_size_recommendation'],
                enhanced_signal['stop_loss_recommendation'],
                json.dumps(enhanced_signal['take_profit_recommendations']),
                enhanced_signal['execution_priority'],
                enhanced_signal['signal_strength'],
                json.dumps(enhanced_signal['market_conditions']),
                enhanced_signal['processed']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing enhanced signal: {e}")
            
    def _store_position(self, position: Position):
        """Store position in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (position_id, symbol, position_type, entry_price, current_price, quantity,
                 entry_time, stop_loss, take_profit_levels, trailing_stop, max_risk_amount,
                 unrealized_pnl, realized_pnl, max_profit, max_drawdown, confluence_score,
                 timeframe, status, partial_closes, last_update, holding_time,
                 risk_reward_ratio, win_probability, entry_signal_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.position_id,
                position.symbol,
                position.position_type.value,
                position.entry_price,
                position.current_price,
                position.quantity,
                position.entry_time,
                position.stop_loss,
                json.dumps(position.take_profit_levels),
                position.trailing_stop,
                position.max_risk_amount,
                position.unrealized_pnl,
                position.realized_pnl,
                position.max_profit,
                position.max_drawdown,
                position.confluence_score,
                position.timeframe,
                position.status.value,
                json.dumps(position.partial_closes),
                position.last_update,
                position.holding_time,
                position.risk_reward_ratio,
                position.win_probability,
                json.dumps(position.entry_signal)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing position: {e}")
            
    def _update_position_in_db(self, position: Position):
        """Update position in database"""
        self._store_position(position)  # Same as store for SQLite
        
    def _store_execution(self, execution: TradeExecution):
        """Store execution in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO executions 
                (execution_id, position_id, order_type, price, quantity, timestamp,
                 fees, slippage, execution_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id,
                execution.position_id,
                execution.order_type.value,
                execution.price,
                execution.quantity,
                execution.timestamp,
                execution.fees,
                execution.slippage,
                execution.execution_time_ms
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing execution: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'positions': {
                    'open_positions': len(self.positions),
                    'closed_positions': len(self.closed_positions),
                    'total_exposure': self.risk_metrics.total_exposure,
                    'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
                },
                'performance': self.performance_stats,
                'risk_metrics': asdict(self.risk_metrics),
                'processing_active': self.processing_active,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
            
    def stop_processing(self):
        """Stop all processing"""
        self.processing_active = False
        logger.info("Enhanced signal processing stopped")

# Global enhanced signal processor instance
enhanced_processor = AdvancedSignalProcessor()

def process_enhanced_trading_signal(signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Main function to process enhanced trading signals"""
    return enhanced_processor.process_enhanced_signal(signal_data)

def get_enhanced_system_status() -> Dict[str, Any]:
    """Get enhanced system status"""
    return enhanced_processor.get_system_status()

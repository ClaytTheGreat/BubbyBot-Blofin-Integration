"""
Advanced Risk Management and Validation System
Comprehensive risk controls, backtesting, and validation for the AI trading bot
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
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from enhanced_signal_processor import Position, PositionType, PositionStatus, enhanced_processor

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class ValidationResult(Enum):
    """Validation result types"""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    REVIEW_REQUIRED = "review_required"

@dataclass
class RiskLimits:
    """Risk management limits and parameters"""
    # Account-level limits
    max_account_risk: float = 0.02  # 2% max risk per trade
    max_daily_loss: float = 0.05    # 5% max daily loss
    max_weekly_loss: float = 0.10   # 10% max weekly loss
    max_monthly_loss: float = 0.20  # 20% max monthly loss
    max_drawdown: float = 0.15      # 15% max drawdown
    
    # Position-level limits
    max_position_size: float = 0.25  # 25% of account per position
    max_leverage: float = 100.0      # Maximum leverage allowed
    min_risk_reward: float = 1.5     # Minimum risk/reward ratio
    max_correlation: float = 0.7     # Maximum correlation between positions
    
    # Time-based limits
    max_trades_per_day: int = 50
    max_trades_per_hour: int = 10
    min_time_between_trades: int = 30  # seconds
    
    # Confidence thresholds
    min_signal_confidence: float = 0.6
    min_confluence_score: float = 0.65
    min_execution_priority: int = 6
    
    # Market condition limits
    max_volatility_threshold: float = 0.8
    min_liquidity_threshold: float = 0.3

@dataclass
class BacktestResult:
    """Backtesting result data structure"""
    strategy_name: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    expected_value: float

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    signal_id: str
    timestamp: str
    validation_result: ValidationResult
    risk_level: RiskLevel
    risk_score: float
    validation_checks: Dict[str, bool]
    risk_factors: List[str]
    recommendations: List[str]
    approved_position_size: float
    approved_leverage: float
    conditions: List[str]

class AdvancedRiskManager:
    """Advanced risk management and validation system"""
    
    def __init__(self):
        self.risk_limits = RiskLimits()
        self.validation_history = deque(maxlen=1000)
        self.risk_events = deque(maxlen=500)
        self.backtest_results = {}
        
        # Risk monitoring
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = 100000.0
        
        # Trade tracking
        self.trades_today = 0
        self.trades_this_hour = 0
        self.last_trade_time = None
        
        # Correlation matrix
        self.correlation_matrix = {}
        
        # Database
        self.db_path = "risk_management.db"
        self.init_database()
        
        # Start monitoring
        self.monitoring_active = True
        threading.Thread(target=self._risk_monitor, daemon=True).start()
        
    def init_database(self):
        """Initialize risk management database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Validation reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_reports (
                signal_id TEXT PRIMARY KEY,
                timestamp TEXT,
                validation_result TEXT,
                risk_level TEXT,
                risk_score REAL,
                validation_checks TEXT,
                risk_factors TEXT,
                recommendations TEXT,
                approved_position_size REAL,
                approved_leverage REAL,
                conditions TEXT
            )
        ''')
        
        # Risk events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                severity TEXT,
                description TEXT,
                affected_positions TEXT,
                action_taken TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                test_id TEXT PRIMARY KEY,
                strategy_name TEXT,
                start_date TEXT,
                end_date TEXT,
                parameters TEXT,
                results TEXT,
                created_at TEXT
            )
        ''')
        
        # Risk metrics history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics_history (
                timestamp TEXT PRIMARY KEY,
                account_balance REAL,
                total_exposure REAL,
                current_drawdown REAL,
                daily_pnl REAL,
                weekly_pnl REAL,
                monthly_pnl REAL,
                var_95 REAL,
                sharpe_ratio REAL,
                active_positions INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def validate_signal(self, enhanced_signal: Dict[str, Any], 
                       position_size: float, leverage: float = 1.0) -> ValidationReport:
        """Comprehensive signal validation"""
        try:
            signal_id = enhanced_signal.get('signal_id', 'unknown')
            timestamp = datetime.now().isoformat()
            
            # Initialize validation checks
            validation_checks = {}
            risk_factors = []
            recommendations = []
            conditions = []
            
            # 1. Signal Quality Validation
            confidence_check = self._validate_signal_confidence(enhanced_signal)
            validation_checks['signal_confidence'] = confidence_check['passed']
            if not confidence_check['passed']:
                risk_factors.extend(confidence_check['issues'])
                
            # 2. Risk/Reward Validation
            rr_check = self._validate_risk_reward(enhanced_signal)
            validation_checks['risk_reward'] = rr_check['passed']
            if not rr_check['passed']:
                risk_factors.extend(rr_check['issues'])
                
            # 3. Position Size Validation
            size_check = self._validate_position_size(enhanced_signal, position_size)
            validation_checks['position_size'] = size_check['passed']
            if not size_check['passed']:
                risk_factors.extend(size_check['issues'])
                
            # 4. Leverage Validation
            leverage_check = self._validate_leverage(enhanced_signal, leverage, position_size)
            validation_checks['leverage'] = leverage_check['passed']
            if not leverage_check['passed']:
                risk_factors.extend(leverage_check['issues'])
                
            # 5. Account Risk Validation
            account_check = self._validate_account_risk(enhanced_signal, position_size, leverage)
            validation_checks['account_risk'] = account_check['passed']
            if not account_check['passed']:
                risk_factors.extend(account_check['issues'])
                
            # 6. Correlation Validation
            correlation_check = self._validate_correlation(enhanced_signal)
            validation_checks['correlation'] = correlation_check['passed']
            if not correlation_check['passed']:
                risk_factors.extend(correlation_check['issues'])
                
            # 7. Market Conditions Validation
            market_check = self._validate_market_conditions(enhanced_signal)
            validation_checks['market_conditions'] = market_check['passed']
            if not market_check['passed']:
                risk_factors.extend(market_check['issues'])
                
            # 8. Time-based Validation
            time_check = self._validate_timing(enhanced_signal)
            validation_checks['timing'] = time_check['passed']
            if not time_check['passed']:
                risk_factors.extend(time_check['issues'])
                
            # 9. Drawdown Validation
            drawdown_check = self._validate_drawdown()
            validation_checks['drawdown'] = drawdown_check['passed']
            if not drawdown_check['passed']:
                risk_factors.extend(drawdown_check['issues'])
                
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(validation_checks, enhanced_signal)
            risk_level = self._determine_risk_level(risk_score)
            
            # Make validation decision
            validation_result, approved_size, approved_leverage = self._make_validation_decision(
                validation_checks, risk_score, position_size, leverage, enhanced_signal
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                validation_checks, risk_factors, enhanced_signal, risk_score
            )
            
            # Generate conditions for conditional approval
            if validation_result == ValidationResult.CONDITIONAL:
                conditions = self._generate_conditions(validation_checks, risk_factors)
            
            # Create validation report
            report = ValidationReport(
                signal_id=signal_id,
                timestamp=timestamp,
                validation_result=validation_result,
                risk_level=risk_level,
                risk_score=risk_score,
                validation_checks=validation_checks,
                risk_factors=risk_factors,
                recommendations=recommendations,
                approved_position_size=approved_size,
                approved_leverage=approved_leverage,
                conditions=conditions
            )
            
            # Store validation report
            self._store_validation_report(report)
            self.validation_history.append(report)
            
            logger.info(f"Signal validation completed: {signal_id} - "
                       f"Result: {validation_result.value} - Risk: {risk_level.value}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in signal validation: {e}")
            return ValidationReport(
                signal_id=signal_id,
                timestamp=datetime.now().isoformat(),
                validation_result=ValidationResult.REJECTED,
                risk_level=RiskLevel.EXTREME,
                risk_score=1.0,
                validation_checks={},
                risk_factors=[f"Validation error: {str(e)}"],
                recommendations=["Review system configuration"],
                approved_position_size=0.0,
                approved_leverage=1.0,
                conditions=[]
            )
            
    def _validate_signal_confidence(self, enhanced_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal confidence levels"""
        try:
            confidence = enhanced_signal.get('confidence', 0.0)
            confluence_score = enhanced_signal.get('confluence_score', 0.0)
            execution_priority = enhanced_signal.get('execution_priority', 0)
            
            issues = []
            
            if confidence < self.risk_limits.min_signal_confidence:
                issues.append(f"Signal confidence {confidence:.3f} below minimum {self.risk_limits.min_signal_confidence}")
                
            if confluence_score < self.risk_limits.min_confluence_score:
                issues.append(f"Confluence score {confluence_score:.3f} below minimum {self.risk_limits.min_confluence_score}")
                
            if execution_priority < self.risk_limits.min_execution_priority:
                issues.append(f"Execution priority {execution_priority} below minimum {self.risk_limits.min_execution_priority}")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'confidence': confidence,
                'confluence_score': confluence_score,
                'execution_priority': execution_priority
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Confidence validation error: {e}"]}
            
    def _validate_risk_reward(self, enhanced_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk/reward ratio"""
        try:
            risk_reward = enhanced_signal.get('risk_reward_ratio', 0.0)
            
            issues = []
            
            if risk_reward < self.risk_limits.min_risk_reward:
                issues.append(f"Risk/reward ratio {risk_reward:.2f} below minimum {self.risk_limits.min_risk_reward}")
            
            # Additional R:R validation based on signal strength
            signal_strength = enhanced_signal.get('signal_strength', 0.5)
            min_rr_for_strength = 1.0 + signal_strength  # Higher strength allows lower R:R
            
            if risk_reward < min_rr_for_strength:
                issues.append(f"Risk/reward {risk_reward:.2f} insufficient for signal strength {signal_strength:.2f}")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'risk_reward': risk_reward,
                'min_required': min_rr_for_strength
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Risk/reward validation error: {e}"]}
            
    def _validate_position_size(self, enhanced_signal: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Validate position size limits"""
        try:
            price = enhanced_signal.get('price', 0.0)
            account_balance = enhanced_processor.risk_metrics.account_balance
            
            position_value = position_size * price
            position_percentage = position_value / account_balance
            
            issues = []
            
            # Check maximum position size (but allow override for high confidence trades)
            confidence = enhanced_signal.get('confidence', 0.0)
            signal_strength = enhanced_signal.get('signal_strength', 0.0)
            
            # Dynamic position size limit based on confidence and strength
            if confidence >= 0.9 and signal_strength >= 0.8:
                # High confidence trades can use larger position sizes
                max_allowed = min(0.5, self.risk_limits.max_position_size * 2)
            elif confidence >= 0.8 and signal_strength >= 0.7:
                max_allowed = min(0.4, self.risk_limits.max_position_size * 1.5)
            else:
                max_allowed = self.risk_limits.max_position_size
            
            if position_percentage > max_allowed:
                issues.append(f"Position size {position_percentage:.2%} exceeds maximum {max_allowed:.2%}")
            
            # Check minimum position size
            if position_value < 100:  # Minimum $100 position
                issues.append(f"Position value ${position_value:.2f} below minimum $100")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'position_percentage': position_percentage,
                'max_allowed': max_allowed,
                'position_value': position_value
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Position size validation error: {e}"]}
            
    def _validate_leverage(self, enhanced_signal: Dict[str, Any], leverage: float, position_size: float) -> Dict[str, Any]:
        """Validate leverage usage with dynamic limits"""
        try:
            confidence = enhanced_signal.get('confidence', 0.0)
            signal_strength = enhanced_signal.get('signal_strength', 0.0)
            stop_loss = enhanced_signal.get('stop_loss_recommendation', 0.0)
            price = enhanced_signal.get('price', 0.0)
            
            issues = []
            
            # Calculate stop loss distance
            if stop_loss > 0 and price > 0:
                stop_distance = abs(price - stop_loss) / price
            else:
                stop_distance = 0.02  # Default 2% stop
            
            # Dynamic leverage limits based on confidence, strength, and stop distance
            if confidence >= 0.9 and signal_strength >= 0.8 and stop_distance <= 0.01:
                # Very high confidence with tight stop: no leverage limit
                max_leverage = self.risk_limits.max_leverage
            elif confidence >= 0.8 and signal_strength >= 0.7 and stop_distance <= 0.02:
                # High confidence with reasonable stop: high leverage allowed
                max_leverage = min(50.0, self.risk_limits.max_leverage)
            elif confidence >= 0.7 and signal_strength >= 0.6 and stop_distance <= 0.03:
                # Medium-high confidence: moderate leverage
                max_leverage = min(20.0, self.risk_limits.max_leverage)
            else:
                # Lower confidence: conservative leverage
                max_leverage = min(10.0, self.risk_limits.max_leverage)
            
            if leverage > max_leverage:
                issues.append(f"Leverage {leverage:.1f}x exceeds maximum {max_leverage:.1f}x for this signal quality")
            
            # Risk-based leverage validation
            account_balance = enhanced_processor.risk_metrics.account_balance
            position_value = position_size * price
            leveraged_exposure = position_value * leverage
            
            if leveraged_exposure > account_balance * 5:  # Max 5x account exposure
                issues.append(f"Leveraged exposure ${leveraged_exposure:.0f} exceeds 5x account balance")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'max_leverage': max_leverage,
                'stop_distance': stop_distance,
                'leveraged_exposure': leveraged_exposure
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Leverage validation error: {e}"]}
            
    def _validate_account_risk(self, enhanced_signal: Dict[str, Any], position_size: float, leverage: float) -> Dict[str, Any]:
        """Validate account-level risk"""
        try:
            price = enhanced_signal.get('price', 0.0)
            stop_loss = enhanced_signal.get('stop_loss_recommendation', price * 0.98)
            account_balance = enhanced_processor.risk_metrics.account_balance
            
            # Calculate risk amount
            risk_per_unit = abs(price - stop_loss) / price if price > 0 else 0.02
            risk_amount = position_size * price * risk_per_unit * leverage
            risk_percentage = risk_amount / account_balance
            
            issues = []
            
            # Dynamic risk limits based on signal quality
            confidence = enhanced_signal.get('confidence', 0.0)
            signal_strength = enhanced_signal.get('signal_strength', 0.0)
            
            if confidence >= 0.9 and signal_strength >= 0.8:
                max_risk = min(0.05, self.risk_limits.max_account_risk * 2.5)  # Up to 5% for exceptional signals
            elif confidence >= 0.8 and signal_strength >= 0.7:
                max_risk = min(0.03, self.risk_limits.max_account_risk * 1.5)  # Up to 3% for strong signals
            else:
                max_risk = self.risk_limits.max_account_risk  # Standard 2% limit
            
            if risk_percentage > max_risk:
                issues.append(f"Account risk {risk_percentage:.2%} exceeds maximum {max_risk:.2%}")
            
            # Check daily loss limits
            if self.daily_pnl < 0 and abs(self.daily_pnl) + risk_amount > account_balance * self.risk_limits.max_daily_loss:
                issues.append(f"Trade would exceed daily loss limit")
            
            # Check weekly loss limits
            if self.weekly_pnl < 0 and abs(self.weekly_pnl) + risk_amount > account_balance * self.risk_limits.max_weekly_loss:
                issues.append(f"Trade would exceed weekly loss limit")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'max_risk': max_risk
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Account risk validation error: {e}"]}
            
    def _validate_correlation(self, enhanced_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate position correlation"""
        try:
            symbol = enhanced_signal.get('symbol', '')
            
            issues = []
            
            # Check correlation with existing positions
            correlation_risk = enhanced_processor._assess_correlation_risk(symbol)
            
            if correlation_risk > self.risk_limits.max_correlation:
                issues.append(f"Correlation risk {correlation_risk:.2f} exceeds maximum {self.risk_limits.max_correlation}")
            
            # Check for overconcentration in similar assets
            similar_positions = 0
            total_exposure = 0
            
            for position in enhanced_processor.positions.values():
                total_exposure += position.quantity * position.current_price
                
                if self._calculate_symbol_correlation(symbol, position.symbol) > 0.7:
                    similar_positions += 1
            
            if similar_positions >= 3:
                issues.append(f"Too many correlated positions ({similar_positions})")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'correlation_risk': correlation_risk,
                'similar_positions': similar_positions
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Correlation validation error: {e}"]}
            
    def _calculate_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            # Simplified correlation calculation
            # In real implementation, would use historical price data
            
            # Same base currency (e.g., BTC pairs)
            if symbol1.startswith('BTC') and symbol2.startswith('BTC'):
                return 0.8
            elif symbol1.startswith('ETH') and symbol2.startswith('ETH'):
                return 0.8
            
            # Same quote currency
            if symbol1.endswith('USDT') and symbol2.endswith('USDT'):
                return 0.3
            elif symbol1.endswith('BTC') and symbol2.endswith('BTC'):
                return 0.5
            
            # Major crypto pairs
            major_cryptos = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
            symbol1_base = symbol1.replace('USDT', '').replace('BTC', '').replace('ETH', '')
            symbol2_base = symbol2.replace('USDT', '').replace('BTC', '').replace('ETH', '')
            
            if symbol1_base in major_cryptos and symbol2_base in major_cryptos:
                return 0.6
            
            return 0.2  # Default low correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.5
            
    def _validate_market_conditions(self, enhanced_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market conditions"""
        try:
            market_conditions = enhanced_signal.get('market_conditions', {})
            
            issues = []
            
            # Check volatility
            volatility = market_conditions.get('volatility', 0.5)
            if volatility > self.risk_limits.max_volatility_threshold:
                issues.append(f"Market volatility {volatility:.2f} exceeds threshold {self.risk_limits.max_volatility_threshold}")
            
            # Check liquidity
            liquidity_score = 0.8 if market_conditions.get('liquidity') == 'high' else 0.5 if market_conditions.get('liquidity') == 'medium' else 0.2
            if liquidity_score < self.risk_limits.min_liquidity_threshold:
                issues.append(f"Market liquidity too low for safe execution")
            
            # Check market session
            session = market_conditions.get('market_session', 'unknown')
            if session == 'asian' and volatility > 0.6:
                issues.append(f"High volatility during low-liquidity Asian session")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'volatility': volatility,
                'liquidity_score': liquidity_score,
                'session': session
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Market conditions validation error: {e}"]}
            
    def _validate_timing(self, enhanced_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate timing constraints"""
        try:
            current_time = datetime.now()
            
            issues = []
            
            # Check daily trade limit
            if self.trades_today >= self.risk_limits.max_trades_per_day:
                issues.append(f"Daily trade limit ({self.risk_limits.max_trades_per_day}) exceeded")
            
            # Check hourly trade limit
            if self.trades_this_hour >= self.risk_limits.max_trades_per_hour:
                issues.append(f"Hourly trade limit ({self.risk_limits.max_trades_per_hour}) exceeded")
            
            # Check minimum time between trades
            if self.last_trade_time:
                time_since_last = (current_time - self.last_trade_time).total_seconds()
                if time_since_last < self.risk_limits.min_time_between_trades:
                    issues.append(f"Minimum time between trades not met ({time_since_last:.0f}s < {self.risk_limits.min_time_between_trades}s)")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'trades_today': self.trades_today,
                'trades_this_hour': self.trades_this_hour,
                'time_since_last': (current_time - self.last_trade_time).total_seconds() if self.last_trade_time else None
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Timing validation error: {e}"]}
            
    def _validate_drawdown(self) -> Dict[str, Any]:
        """Validate current drawdown levels"""
        try:
            issues = []
            
            if self.current_drawdown > self.risk_limits.max_drawdown:
                issues.append(f"Current drawdown {self.current_drawdown:.2%} exceeds maximum {self.risk_limits.max_drawdown:.2%}")
            
            # Check if approaching drawdown limit
            if self.current_drawdown > self.risk_limits.max_drawdown * 0.8:
                issues.append(f"Approaching maximum drawdown limit")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.risk_limits.max_drawdown
            }
            
        except Exception as e:
            return {'passed': False, 'issues': [f"Drawdown validation error: {e}"]}
            
    def _calculate_risk_score(self, validation_checks: Dict[str, bool], enhanced_signal: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-1, higher = more risky)"""
        try:
            # Base risk from failed checks
            failed_checks = sum(1 for passed in validation_checks.values() if not passed)
            total_checks = len(validation_checks)
            
            base_risk = failed_checks / total_checks if total_checks > 0 else 0.5
            
            # Adjust for signal quality
            confidence = enhanced_signal.get('confidence', 0.5)
            signal_strength = enhanced_signal.get('signal_strength', 0.5)
            
            # Lower confidence/strength increases risk
            quality_risk = (2 - confidence - signal_strength) / 2
            
            # Market conditions risk
            market_conditions = enhanced_signal.get('market_conditions', {})
            volatility = market_conditions.get('volatility', 0.5)
            
            # Combine risk factors
            overall_risk = (base_risk * 0.5 + quality_risk * 0.3 + volatility * 0.2)
            
            return min(1.0, max(0.0, overall_risk))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.8  # Conservative default
            
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score <= 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score <= 0.4:
            return RiskLevel.LOW
        elif risk_score <= 0.6:
            return RiskLevel.MEDIUM
        elif risk_score <= 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
            
    def _make_validation_decision(self, validation_checks: Dict[str, bool], risk_score: float,
                                position_size: float, leverage: float, enhanced_signal: Dict[str, Any]) -> Tuple[ValidationResult, float, float]:
        """Make final validation decision"""
        try:
            failed_checks = [check for check, passed in validation_checks.items() if not passed]
            critical_failures = ['account_risk', 'drawdown', 'leverage']
            
            # Check for critical failures
            if any(check in critical_failures for check in failed_checks):
                return ValidationResult.REJECTED, 0.0, 1.0
            
            # Check risk score
            if risk_score > 0.8:
                return ValidationResult.REJECTED, 0.0, 1.0
            elif risk_score > 0.6:
                # Conditional approval with reduced size/leverage
                approved_size = position_size * 0.5
                approved_leverage = min(leverage, 5.0)
                return ValidationResult.CONDITIONAL, approved_size, approved_leverage
            elif risk_score > 0.4:
                # Conditional approval with slight reduction
                approved_size = position_size * 0.8
                approved_leverage = min(leverage, 10.0)
                return ValidationResult.CONDITIONAL, approved_size, approved_leverage
            else:
                # Full approval
                return ValidationResult.APPROVED, position_size, leverage
                
        except Exception as e:
            logger.error(f"Error making validation decision: {e}")
            return ValidationResult.REJECTED, 0.0, 1.0
            
    def _generate_recommendations(self, validation_checks: Dict[str, bool], risk_factors: List[str],
                                enhanced_signal: Dict[str, Any], risk_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            if risk_score > 0.6:
                recommendations.append("Consider reducing position size due to elevated risk")
            
            if not validation_checks.get('signal_confidence', True):
                recommendations.append("Wait for higher confidence signal")
            
            if not validation_checks.get('risk_reward', True):
                recommendations.append("Look for better risk/reward opportunities")
            
            if not validation_checks.get('correlation', True):
                recommendations.append("Reduce exposure to correlated assets")
            
            if not validation_checks.get('market_conditions', True):
                recommendations.append("Wait for better market conditions")
            
            if not validation_checks.get('timing', True):
                recommendations.append("Respect trading frequency limits")
            
            # Signal-specific recommendations
            confidence = enhanced_signal.get('confidence', 0.5)
            if confidence < 0.7:
                recommendations.append("Consider paper trading this signal first")
            
            market_conditions = enhanced_signal.get('market_conditions', {})
            if market_conditions.get('volatility', 0.5) > 0.7:
                recommendations.append("Use tighter stop losses in high volatility")
            
            if len(recommendations) == 0:
                recommendations.append("Signal meets all risk criteria - proceed with caution")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Review risk parameters manually")
        
        return recommendations
        
    def _generate_conditions(self, validation_checks: Dict[str, bool], risk_factors: List[str]) -> List[str]:
        """Generate conditions for conditional approval"""
        conditions = []
        
        try:
            if not validation_checks.get('position_size', True):
                conditions.append("Reduce position size by 50%")
            
            if not validation_checks.get('leverage', True):
                conditions.append("Limit leverage to 5x maximum")
            
            if not validation_checks.get('correlation', True):
                conditions.append("Close one correlated position first")
            
            if not validation_checks.get('market_conditions', True):
                conditions.append("Use tighter stop loss due to market conditions")
            
            if not validation_checks.get('timing', True):
                conditions.append("Wait for minimum time between trades")
                
        except Exception as e:
            logger.error(f"Error generating conditions: {e}")
            conditions.append("Manual review required")
        
        return conditions
        
    def run_backtest(self, strategy_name: str, start_date: str, end_date: str,
                    parameters: Dict[str, Any] = None) -> BacktestResult:
        """Run comprehensive backtesting"""
        try:
            logger.info(f"Starting backtest: {strategy_name} from {start_date} to {end_date}")
            
            # Simulate backtesting (in real implementation, would use historical data)
            np.random.seed(42)  # For reproducible results
            
            # Generate simulated trade results
            num_trades = np.random.randint(100, 500)
            
            # Simulate win rate based on strategy parameters
            base_win_rate = 0.6
            if parameters:
                confidence_threshold = parameters.get('min_confidence', 0.6)
                base_win_rate += (confidence_threshold - 0.6) * 0.5
            
            wins = np.random.binomial(num_trades, base_win_rate)
            losses = num_trades - wins
            
            # Generate P&L distribution
            win_amounts = np.random.lognormal(0.5, 0.8, wins)  # Positive skew for wins
            loss_amounts = -np.random.exponential(0.8, losses)  # Exponential for losses
            
            all_trades = np.concatenate([win_amounts, loss_amounts])
            np.random.shuffle(all_trades)
            
            # Calculate metrics
            total_return = np.sum(all_trades)
            win_rate = wins / num_trades
            average_win = np.mean(win_amounts) if wins > 0 else 0
            average_loss = np.mean(loss_amounts) if losses > 0 else 0
            largest_win = np.max(win_amounts) if wins > 0 else 0
            largest_loss = np.min(loss_amounts) if losses > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(all_trades)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / np.maximum(running_max, 1)
            max_drawdown = np.max(drawdowns)
            
            # Calculate Sharpe ratio
            if np.std(all_trades) > 0:
                sharpe_ratio = np.mean(all_trades) / np.std(all_trades) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate profit factor
            total_wins = np.sum(win_amounts) if wins > 0 else 0
            total_losses = abs(np.sum(loss_amounts)) if losses > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Calculate consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak = 0
            
            for trade in all_trades:
                if trade > 0:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
            
            # Calculate additional metrics
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = all_trades[all_trades < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.01
            sortino_ratio = np.mean(all_trades) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Value at Risk (95%)
            var_95 = np.percentile(all_trades, 5) if len(all_trades) > 0 else 0
            
            # Expected value
            expected_value = np.mean(all_trades)
            
            # Create backtest result
            result = BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                total_trades=num_trades,
                winning_trades=wins,
                losing_trades=losses,
                win_rate=win_rate,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=max_consecutive_wins,
                consecutive_losses=max_consecutive_losses,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                expected_value=expected_value
            )
            
            # Store backtest result
            self._store_backtest_result(result, parameters)
            self.backtest_results[strategy_name] = result
            
            logger.info(f"Backtest completed: {strategy_name} - "
                       f"Win Rate: {win_rate:.2%} - Total Return: {total_return:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
            
    def _risk_monitor(self):
        """Continuous risk monitoring"""
        while self.monitoring_active:
            try:
                self._update_risk_metrics()
                self._check_risk_alerts()
                self._update_trade_counters()
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                time.sleep(300)
                
    def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            # Get current positions
            positions = enhanced_processor.positions
            closed_positions = enhanced_processor.closed_positions
            
            # Calculate current balance
            current_balance = enhanced_processor.risk_metrics.account_balance
            total_unrealized = sum(pos.unrealized_pnl for pos in positions.values())
            
            # Update peak balance
            current_equity = current_balance + total_unrealized
            if current_equity > self.peak_balance:
                self.peak_balance = current_equity
            
            # Calculate drawdown
            self.current_drawdown = (self.peak_balance - current_equity) / self.peak_balance
            
            # Update P&L tracking
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
            month_start = today.replace(day=1)
            
            # Calculate daily P&L
            daily_trades = [pos for pos in closed_positions 
                          if datetime.fromisoformat(pos.last_update).date() == today]
            self.daily_pnl = sum(pos.realized_pnl for pos in daily_trades)
            
            # Calculate weekly P&L
            weekly_trades = [pos for pos in closed_positions 
                           if datetime.fromisoformat(pos.last_update).date() >= week_start]
            self.weekly_pnl = sum(pos.realized_pnl for pos in weekly_trades)
            
            # Calculate monthly P&L
            monthly_trades = [pos for pos in closed_positions 
                            if datetime.fromisoformat(pos.last_update).date() >= month_start]
            self.monthly_pnl = sum(pos.realized_pnl for pos in monthly_trades)
            
            # Store metrics in database
            self._store_risk_metrics()
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            
    def _check_risk_alerts(self):
        """Check for risk alerts and events"""
        try:
            alerts = []
            
            # Drawdown alerts
            if self.current_drawdown > self.risk_limits.max_drawdown * 0.8:
                alerts.append({
                    'type': 'drawdown_warning',
                    'severity': 'high',
                    'message': f"Approaching maximum drawdown: {self.current_drawdown:.2%}"
                })
            
            if self.current_drawdown > self.risk_limits.max_drawdown:
                alerts.append({
                    'type': 'drawdown_exceeded',
                    'severity': 'critical',
                    'message': f"Maximum drawdown exceeded: {self.current_drawdown:.2%}"
                })
            
            # Daily loss alerts
            account_balance = enhanced_processor.risk_metrics.account_balance
            if self.daily_pnl < -account_balance * self.risk_limits.max_daily_loss * 0.8:
                alerts.append({
                    'type': 'daily_loss_warning',
                    'severity': 'medium',
                    'message': f"Approaching daily loss limit: {self.daily_pnl:.2f}"
                })
            
            # Process alerts
            for alert in alerts:
                self._process_risk_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
            
    def _process_risk_alert(self, alert: Dict[str, Any]):
        """Process risk alert"""
        try:
            event_id = f"risk_{int(time.time() * 1000)}"
            
            # Log alert
            logger.warning(f"Risk Alert: {alert['type']} - {alert['message']}")
            
            # Store risk event
            self._store_risk_event(event_id, alert)
            
            # Take action based on severity
            if alert['severity'] == 'critical':
                # Could implement automatic position closure here
                logger.critical(f"Critical risk event: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error processing risk alert: {e}")
            
    def _update_trade_counters(self):
        """Update trade frequency counters"""
        try:
            current_time = datetime.now()
            
            # Reset daily counter at midnight
            if hasattr(self, '_last_day') and current_time.date() != self._last_day:
                self.trades_today = 0
            self._last_day = current_time.date()
            
            # Reset hourly counter
            if hasattr(self, '_last_hour') and current_time.hour != self._last_hour:
                self.trades_this_hour = 0
            self._last_hour = current_time.hour
            
        except Exception as e:
            logger.error(f"Error updating trade counters: {e}")
            
    def record_trade_execution(self):
        """Record a trade execution for frequency tracking"""
        try:
            self.trades_today += 1
            self.trades_this_hour += 1
            self.last_trade_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error recording trade execution: {e}")
            
    def _store_validation_report(self, report: ValidationReport):
        """Store validation report in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO validation_reports 
                (signal_id, timestamp, validation_result, risk_level, risk_score,
                 validation_checks, risk_factors, recommendations, approved_position_size,
                 approved_leverage, conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.signal_id,
                report.timestamp,
                report.validation_result.value,
                report.risk_level.value,
                report.risk_score,
                json.dumps(report.validation_checks),
                json.dumps(report.risk_factors),
                json.dumps(report.recommendations),
                report.approved_position_size,
                report.approved_leverage,
                json.dumps(report.conditions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing validation report: {e}")
            
    def _store_backtest_result(self, result: BacktestResult, parameters: Dict[str, Any]):
        """Store backtest result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            test_id = f"backtest_{int(time.time() * 1000)}"
            
            cursor.execute('''
                INSERT INTO backtest_results 
                (test_id, strategy_name, start_date, end_date, parameters, results, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_id,
                result.strategy_name,
                result.start_date,
                result.end_date,
                json.dumps(parameters or {}),
                json.dumps(asdict(result)),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing backtest result: {e}")
            
    def _store_risk_event(self, event_id: str, alert: Dict[str, Any]):
        """Store risk event in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_events 
                (event_id, timestamp, event_type, severity, description, 
                 affected_positions, action_taken, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                datetime.now().isoformat(),
                alert['type'],
                alert['severity'],
                alert['message'],
                json.dumps([]),  # Could list affected positions
                json.dumps([]),  # Could list actions taken
                False
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing risk event: {e}")
            
    def _store_risk_metrics(self):
        """Store current risk metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_metrics_history 
                (timestamp, account_balance, total_exposure, current_drawdown,
                 daily_pnl, weekly_pnl, monthly_pnl, var_95, sharpe_ratio, active_positions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                enhanced_processor.risk_metrics.account_balance,
                enhanced_processor.risk_metrics.total_exposure,
                self.current_drawdown,
                self.daily_pnl,
                self.weekly_pnl,
                self.monthly_pnl,
                0.0,  # VaR calculation would go here
                0.0,  # Sharpe ratio calculation would go here
                len(enhanced_processor.positions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
            
    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status"""
        try:
            return {
                'risk_limits': asdict(self.risk_limits),
                'current_metrics': {
                    'current_drawdown': self.current_drawdown,
                    'daily_pnl': self.daily_pnl,
                    'weekly_pnl': self.weekly_pnl,
                    'monthly_pnl': self.monthly_pnl,
                    'peak_balance': self.peak_balance,
                    'trades_today': self.trades_today,
                    'trades_this_hour': self.trades_this_hour
                },
                'validation_stats': {
                    'total_validations': len(self.validation_history),
                    'recent_approvals': sum(1 for v in list(self.validation_history)[-10:] 
                                          if v.validation_result == ValidationResult.APPROVED),
                    'recent_rejections': sum(1 for v in list(self.validation_history)[-10:] 
                                           if v.validation_result == ValidationResult.REJECTED)
                },
                'monitoring_active': self.monitoring_active,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {}
            
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
        logger.info("Risk monitoring stopped")

# Global risk manager instance
risk_manager = AdvancedRiskManager()

def validate_trading_signal(enhanced_signal: Dict[str, Any], position_size: float, 
                          leverage: float = 1.0) -> ValidationReport:
    """Main function to validate trading signals"""
    return risk_manager.validate_signal(enhanced_signal, position_size, leverage)

def run_strategy_backtest(strategy_name: str, start_date: str, end_date: str,
                         parameters: Dict[str, Any] = None) -> BacktestResult:
    """Main function to run backtests"""
    return risk_manager.run_backtest(strategy_name, start_date, end_date, parameters)

def get_risk_management_status() -> Dict[str, Any]:
    """Get risk management status"""
    return risk_manager.get_risk_status()

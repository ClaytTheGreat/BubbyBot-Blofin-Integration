"""
Advanced Multi-Timeframe Analysis Engine
Comprehensive learning system from 1-second to monthly timeframes
"""

import logging
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """All supported timeframes from 1 second to monthly"""
    S1 = "1s"      # 1 second - Ultra high frequency
    S5 = "5s"      # 5 seconds
    S15 = "15s"    # 15 seconds
    S30 = "30s"    # 30 seconds
    M1 = "1m"      # 1 minute - Scalping
    M3 = "3m"      # 3 minutes
    M5 = "5m"      # 5 minutes - Short-term
    M15 = "15m"    # 15 minutes
    M30 = "30m"    # 30 minutes
    H1 = "1h"      # 1 hour - Day trading
    H2 = "2h"      # 2 hours
    H4 = "4h"      # 4 hours - Swing trading
    H6 = "6h"      # 6 hours
    H12 = "12h"    # 12 hours
    D1 = "1D"      # Daily - Position trading
    D3 = "3D"      # 3 days
    W1 = "1W"      # Weekly - Long-term
    MN1 = "1M"     # Monthly - Macro analysis

class TradingStyle(Enum):
    """Trading styles based on timeframes"""
    SCALPING = "scalping"           # 1s-1m
    DAY_TRADING = "day_trading"     # 5m-4h
    SWING_TRADING = "swing_trading" # 4h-1D
    POSITION_TRADING = "position"   # 1D-1M

@dataclass
class TimeframeData:
    """Data structure for each timeframe analysis"""
    timeframe: str
    timestamp: str
    price: float
    volume: float
    
    # Market Cipher signals
    mc_a_trend: str
    mc_a_strength: float
    mc_b_money_flow: str
    mc_b_momentum: float
    mc_sr_support: float
    mc_sr_resistance: float
    mc_dbsi_score: float
    
    # Lux Algo signals
    la_order_block_type: str
    la_order_block_strength: float
    la_premium_discount_zone: str
    la_market_structure: str
    
    # Pattern recognition
    detected_patterns: List[str]
    confluence_score: float
    volatility: float
    trend_strength: float

@dataclass
class HistoricalPattern:
    """Historical pattern for learning"""
    pattern_id: str
    timeframes: List[str]
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    success_rate: float
    avg_profit: float
    avg_loss: float
    risk_reward_ratio: float
    frequency: int
    last_seen: str

class PatternDatabase:
    """Database for storing and analyzing historical patterns"""
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for pattern storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE,
                timeframes TEXT,
                entry_conditions TEXT,
                exit_conditions TEXT,
                success_rate REAL,
                avg_profit REAL,
                avg_loss REAL,
                risk_reward_ratio REAL,
                frequency INTEGER,
                last_seen TEXT,
                created_at TEXT
            )
        ''')
        
        # Create trades table for backtesting
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT,
                timeframe TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                profit_loss REAL,
                success INTEGER,
                confluence_score REAL,
                created_at TEXT
            )
        ''')
        
        # Create market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timeframe TEXT,
                timestamp TEXT,
                symbol TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                mc_data TEXT,
                la_data TEXT,
                patterns TEXT,
                confluence_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def store_pattern(self, pattern: HistoricalPattern):
        """Store or update a historical pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO patterns 
            (pattern_id, timeframes, entry_conditions, exit_conditions, 
             success_rate, avg_profit, avg_loss, risk_reward_ratio, 
             frequency, last_seen, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            json.dumps(pattern.timeframes),
            json.dumps(pattern.entry_conditions),
            json.dumps(pattern.exit_conditions),
            pattern.success_rate,
            pattern.avg_profit,
            pattern.avg_loss,
            pattern.risk_reward_ratio,
            pattern.frequency,
            pattern.last_seen,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def get_patterns_by_timeframe(self, timeframe: str) -> List[HistoricalPattern]:
        """Get patterns for specific timeframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM patterns WHERE timeframes LIKE ?
        ''', (f'%{timeframe}%',))
        
        patterns = []
        for row in cursor.fetchall():
            pattern = HistoricalPattern(
                pattern_id=row[1],
                timeframes=json.loads(row[2]),
                entry_conditions=json.loads(row[3]),
                exit_conditions=json.loads(row[4]),
                success_rate=row[5],
                avg_profit=row[6],
                avg_loss=row[7],
                risk_reward_ratio=row[8],
                frequency=row[9],
                last_seen=row[10]
            )
            patterns.append(pattern)
        
        conn.close()
        return patterns

class MultiTimeframeAnalyzer:
    """Advanced multi-timeframe analysis engine"""
    
    def __init__(self):
        self.pattern_db = PatternDatabase()
        self.timeframe_data = {}
        self.learning_active = False
        self.analysis_results = defaultdict(dict)
        
        # Timeframe weights for different trading styles
        self.timeframe_weights = {
            TradingStyle.SCALPING: {
                Timeframe.S1: 0.25, Timeframe.S5: 0.20, Timeframe.S15: 0.15,
                Timeframe.S30: 0.15, Timeframe.M1: 0.15, Timeframe.M3: 0.10
            },
            TradingStyle.DAY_TRADING: {
                Timeframe.M1: 0.10, Timeframe.M5: 0.20, Timeframe.M15: 0.25,
                Timeframe.M30: 0.20, Timeframe.H1: 0.15, Timeframe.H4: 0.10
            },
            TradingStyle.SWING_TRADING: {
                Timeframe.H1: 0.15, Timeframe.H4: 0.30, Timeframe.H12: 0.25,
                Timeframe.D1: 0.20, Timeframe.W1: 0.10
            },
            TradingStyle.POSITION_TRADING: {
                Timeframe.D1: 0.30, Timeframe.D3: 0.25, Timeframe.W1: 0.30,
                Timeframe.MN1: 0.15
            }
        }
        
    def start_continuous_learning(self):
        """Start continuous learning across all timeframes"""
        self.learning_active = True
        
        # Start learning threads for different timeframe groups
        threading.Thread(target=self._learn_scalping_patterns, daemon=True).start()
        threading.Thread(target=self._learn_day_trading_patterns, daemon=True).start()
        threading.Thread(target=self._learn_swing_patterns, daemon=True).start()
        threading.Thread(target=self._learn_position_patterns, daemon=True).start()
        
        # Start pattern analysis thread
        threading.Thread(target=self._analyze_cross_timeframe_patterns, daemon=True).start()
        
        logger.info("Started continuous multi-timeframe learning system")
        
    def _learn_scalping_patterns(self):
        """Learn scalping patterns from 1s to 3m timeframes"""
        while self.learning_active:
            try:
                # Simulate learning from ultra-short timeframes
                timeframes = [Timeframe.S1, Timeframe.S5, Timeframe.S15, Timeframe.S30, Timeframe.M1, Timeframe.M3]
                
                for tf in timeframes:
                    # Analyze micro-patterns
                    patterns = self._detect_micro_patterns(tf)
                    
                    # Store successful patterns
                    for pattern in patterns:
                        if pattern.success_rate > 0.6:  # Only store profitable patterns
                            self.pattern_db.store_pattern(pattern)
                
                # Learn every 10 seconds for scalping
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in scalping pattern learning: {e}")
                time.sleep(30)
                
    def _learn_day_trading_patterns(self):
        """Learn day trading patterns from 5m to 4h timeframes"""
        while self.learning_active:
            try:
                timeframes = [Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]
                
                for tf in timeframes:
                    # Analyze intraday patterns
                    patterns = self._detect_intraday_patterns(tf)
                    
                    # Cross-timeframe confluence analysis
                    confluence_patterns = self._analyze_timeframe_confluence(tf, timeframes)
                    
                    # Store patterns with good performance
                    all_patterns = patterns + confluence_patterns
                    for pattern in all_patterns:
                        if pattern.success_rate > 0.65:
                            self.pattern_db.store_pattern(pattern)
                
                # Learn every 5 minutes for day trading
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in day trading pattern learning: {e}")
                time.sleep(300)
                
    def _learn_swing_patterns(self):
        """Learn swing trading patterns from 4h to weekly timeframes"""
        while self.learning_active:
            try:
                timeframes = [Timeframe.H4, Timeframe.H6, Timeframe.H12, Timeframe.D1, Timeframe.D3, Timeframe.W1]
                
                for tf in timeframes:
                    # Analyze swing patterns
                    patterns = self._detect_swing_patterns(tf)
                    
                    # Multi-day pattern analysis
                    extended_patterns = self._analyze_extended_patterns(tf)
                    
                    # Store high-quality swing patterns
                    all_patterns = patterns + extended_patterns
                    for pattern in all_patterns:
                        if pattern.success_rate > 0.70 and pattern.risk_reward_ratio > 2.0:
                            self.pattern_db.store_pattern(pattern)
                
                # Learn every hour for swing trading
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in swing pattern learning: {e}")
                time.sleep(3600)
                
    def _learn_position_patterns(self):
        """Learn position trading patterns from daily to monthly timeframes"""
        while self.learning_active:
            try:
                timeframes = [Timeframe.D1, Timeframe.D3, Timeframe.W1, Timeframe.MN1]
                
                for tf in timeframes:
                    # Analyze long-term trends and patterns
                    patterns = self._detect_macro_patterns(tf)
                    
                    # Seasonal and cyclical pattern analysis
                    cyclical_patterns = self._analyze_cyclical_patterns(tf)
                    
                    # Store high-conviction long-term patterns
                    all_patterns = patterns + cyclical_patterns
                    for pattern in all_patterns:
                        if pattern.success_rate > 0.75 and pattern.risk_reward_ratio > 3.0:
                            self.pattern_db.store_pattern(pattern)
                
                # Learn every 4 hours for position trading
                time.sleep(14400)
                
            except Exception as e:
                logger.error(f"Error in position pattern learning: {e}")
                time.sleep(14400)
                
    def _detect_micro_patterns(self, timeframe: Timeframe) -> List[HistoricalPattern]:
        """Detect micro-patterns for scalping (1s-3m)"""
        patterns = []
        
        try:
            # Simulate pattern detection for ultra-short timeframes
            pattern_types = [
                "mc_money_flow_spike",
                "la_order_block_touch",
                "volume_surge_breakout",
                "momentum_divergence",
                "squeeze_release"
            ]
            
            for pattern_type in pattern_types:
                pattern = HistoricalPattern(
                    pattern_id=f"{pattern_type}_{timeframe.value}",
                    timeframes=[timeframe.value],
                    entry_conditions={
                        "mc_b_money_flow": "strong_bullish" if "bullish" in pattern_type else "strong_bearish",
                        "la_order_block": "active",
                        "volume_surge": True,
                        "confluence_score": "> 0.7"
                    },
                    exit_conditions={
                        "profit_target": "0.5%",
                        "stop_loss": "0.2%",
                        "time_exit": "30 seconds"
                    },
                    success_rate=np.random.uniform(0.55, 0.75),  # Simulated success rates
                    avg_profit=np.random.uniform(0.3, 0.8),
                    avg_loss=np.random.uniform(0.15, 0.25),
                    risk_reward_ratio=np.random.uniform(1.5, 3.0),
                    frequency=np.random.randint(10, 50),
                    last_seen=datetime.now().isoformat()
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting micro patterns: {e}")
            
        return patterns
        
    def _detect_intraday_patterns(self, timeframe: Timeframe) -> List[HistoricalPattern]:
        """Detect intraday patterns for day trading (5m-4h)"""
        patterns = []
        
        try:
            pattern_types = [
                "morning_breakout",
                "midday_reversal",
                "afternoon_continuation",
                "market_open_gap",
                "session_close_momentum"
            ]
            
            for pattern_type in pattern_types:
                pattern = HistoricalPattern(
                    pattern_id=f"{pattern_type}_{timeframe.value}",
                    timeframes=[timeframe.value],
                    entry_conditions={
                        "time_of_day": "market_hours",
                        "mc_a_trend": "aligned",
                        "mc_dbsi_momentum": "> 0.6",
                        "la_premium_discount": "favorable",
                        "volume_profile": "above_average"
                    },
                    exit_conditions={
                        "profit_target": "1.5%",
                        "stop_loss": "0.8%",
                        "trailing_stop": "0.5%"
                    },
                    success_rate=np.random.uniform(0.60, 0.80),
                    avg_profit=np.random.uniform(1.0, 2.5),
                    avg_loss=np.random.uniform(0.6, 1.0),
                    risk_reward_ratio=np.random.uniform(1.8, 3.5),
                    frequency=np.random.randint(5, 25),
                    last_seen=datetime.now().isoformat()
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting intraday patterns: {e}")
            
        return patterns
        
    def _detect_swing_patterns(self, timeframe: Timeframe) -> List[HistoricalPattern]:
        """Detect swing patterns for swing trading (4h-1W)"""
        patterns = []
        
        try:
            pattern_types = [
                "weekly_trend_continuation",
                "daily_support_bounce",
                "resistance_breakout",
                "fibonacci_retracement",
                "market_structure_shift"
            ]
            
            for pattern_type in pattern_types:
                pattern = HistoricalPattern(
                    pattern_id=f"{pattern_type}_{timeframe.value}",
                    timeframes=[timeframe.value],
                    entry_conditions={
                        "trend_alignment": "multi_timeframe",
                        "mc_sr_levels": "respected",
                        "la_market_structure": "bullish_shift",
                        "confluence_score": "> 0.75",
                        "risk_reward": "> 2.0"
                    },
                    exit_conditions={
                        "profit_target": "4%",
                        "stop_loss": "1.5%",
                        "time_exit": "5 days"
                    },
                    success_rate=np.random.uniform(0.65, 0.85),
                    avg_profit=np.random.uniform(3.0, 6.0),
                    avg_loss=np.random.uniform(1.2, 2.0),
                    risk_reward_ratio=np.random.uniform(2.5, 4.0),
                    frequency=np.random.randint(2, 15),
                    last_seen=datetime.now().isoformat()
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting swing patterns: {e}")
            
        return patterns
        
    def _detect_macro_patterns(self, timeframe: Timeframe) -> List[HistoricalPattern]:
        """Detect macro patterns for position trading (1D-1M)"""
        patterns = []
        
        try:
            pattern_types = [
                "monthly_trend_reversal",
                "weekly_accumulation",
                "seasonal_pattern",
                "macro_breakout",
                "long_term_support_test"
            ]
            
            for pattern_type in pattern_types:
                pattern = HistoricalPattern(
                    pattern_id=f"{pattern_type}_{timeframe.value}",
                    timeframes=[timeframe.value],
                    entry_conditions={
                        "macro_trend": "established",
                        "fundamental_alignment": "positive",
                        "technical_confluence": "> 0.8",
                        "market_sentiment": "favorable",
                        "risk_reward": "> 3.0"
                    },
                    exit_conditions={
                        "profit_target": "15%",
                        "stop_loss": "5%",
                        "time_exit": "3 months"
                    },
                    success_rate=np.random.uniform(0.70, 0.90),
                    avg_profit=np.random.uniform(10.0, 25.0),
                    avg_loss=np.random.uniform(3.0, 6.0),
                    risk_reward_ratio=np.random.uniform(3.0, 5.0),
                    frequency=np.random.randint(1, 8),
                    last_seen=datetime.now().isoformat()
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting macro patterns: {e}")
            
        return patterns
        
    def _analyze_timeframe_confluence(self, primary_tf: Timeframe, all_timeframes: List[Timeframe]) -> List[HistoricalPattern]:
        """Analyze confluence across multiple timeframes"""
        confluence_patterns = []
        
        try:
            # Create multi-timeframe confluence patterns
            pattern = HistoricalPattern(
                pattern_id=f"mtf_confluence_{primary_tf.value}",
                timeframes=[tf.value for tf in all_timeframes],
                entry_conditions={
                    "primary_timeframe": primary_tf.value,
                    "higher_tf_trend": "aligned",
                    "lower_tf_entry": "confirmed",
                    "confluence_score": "> 0.8"
                },
                exit_conditions={
                    "profit_target": "dynamic",
                    "stop_loss": "atr_based",
                    "trailing_stop": "enabled"
                },
                success_rate=np.random.uniform(0.75, 0.90),
                avg_profit=np.random.uniform(2.0, 5.0),
                avg_loss=np.random.uniform(0.8, 1.5),
                risk_reward_ratio=np.random.uniform(2.5, 4.5),
                frequency=np.random.randint(3, 12),
                last_seen=datetime.now().isoformat()
            )
            confluence_patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe confluence: {e}")
            
        return confluence_patterns
        
    def _analyze_extended_patterns(self, timeframe: Timeframe) -> List[HistoricalPattern]:
        """Analyze extended multi-day patterns"""
        patterns = []
        
        try:
            # Extended pattern analysis for swing trading
            extended_types = [
                "three_day_reversal",
                "weekly_consolidation_breakout",
                "gap_fill_continuation",
                "support_resistance_retest"
            ]
            
            for pattern_type in extended_types:
                pattern = HistoricalPattern(
                    pattern_id=f"{pattern_type}_{timeframe.value}",
                    timeframes=[timeframe.value],
                    entry_conditions={
                        "pattern_duration": "3-7 days",
                        "volume_confirmation": "required",
                        "multiple_touches": "support/resistance",
                        "confluence_score": "> 0.7"
                    },
                    exit_conditions={
                        "profit_target": "5%",
                        "stop_loss": "2%",
                        "time_exit": "2 weeks"
                    },
                    success_rate=np.random.uniform(0.68, 0.82),
                    avg_profit=np.random.uniform(4.0, 8.0),
                    avg_loss=np.random.uniform(1.5, 2.5),
                    risk_reward_ratio=np.random.uniform(2.0, 3.5),
                    frequency=np.random.randint(2, 10),
                    last_seen=datetime.now().isoformat()
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error analyzing extended patterns: {e}")
            
        return patterns
        
    def _analyze_cyclical_patterns(self, timeframe: Timeframe) -> List[HistoricalPattern]:
        """Analyze cyclical and seasonal patterns"""
        patterns = []
        
        try:
            # Cyclical pattern analysis for position trading
            cyclical_types = [
                "monthly_cycle",
                "quarterly_earnings_effect",
                "seasonal_trend",
                "market_cycle_phase"
            ]
            
            for pattern_type in cyclical_types:
                pattern = HistoricalPattern(
                    pattern_id=f"{pattern_type}_{timeframe.value}",
                    timeframes=[timeframe.value],
                    entry_conditions={
                        "cycle_phase": "accumulation",
                        "seasonal_factor": "positive",
                        "macro_environment": "supportive",
                        "confluence_score": "> 0.75"
                    },
                    exit_conditions={
                        "profit_target": "20%",
                        "stop_loss": "7%",
                        "cycle_completion": "exit_signal"
                    },
                    success_rate=np.random.uniform(0.72, 0.88),
                    avg_profit=np.random.uniform(15.0, 30.0),
                    avg_loss=np.random.uniform(5.0, 8.0),
                    risk_reward_ratio=np.random.uniform(3.0, 4.5),
                    frequency=np.random.randint(1, 6),
                    last_seen=datetime.now().isoformat()
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error analyzing cyclical patterns: {e}")
            
        return patterns
        
    def _analyze_cross_timeframe_patterns(self):
        """Analyze patterns across all timeframes for optimal strategies"""
        while self.learning_active:
            try:
                # Get patterns from all timeframes
                all_patterns = {}
                for tf in Timeframe:
                    patterns = self.pattern_db.get_patterns_by_timeframe(tf.value)
                    all_patterns[tf.value] = patterns
                
                # Analyze cross-timeframe relationships
                self._find_optimal_entry_combinations(all_patterns)
                self._analyze_timeframe_hierarchies(all_patterns)
                self._optimize_exit_strategies(all_patterns)
                
                # Update analysis results
                self.analysis_results['last_analysis'] = datetime.now().isoformat()
                self.analysis_results['total_patterns'] = sum(len(patterns) for patterns in all_patterns.values())
                
                logger.info(f"Cross-timeframe analysis completed. Total patterns: {self.analysis_results['total_patterns']}")
                
                # Analyze every 30 minutes
                time.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in cross-timeframe analysis: {e}")
                time.sleep(1800)
                
    def _find_optimal_entry_combinations(self, all_patterns: Dict[str, List[HistoricalPattern]]):
        """Find optimal entry combinations across timeframes"""
        try:
            # Analyze which timeframe combinations produce the best results
            combinations = []
            
            # Short-term + Medium-term combinations
            for short_tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
                for medium_tf in [Timeframe.H1, Timeframe.H4]:
                    combo_success = self._calculate_combination_success(
                        all_patterns.get(short_tf.value, []),
                        all_patterns.get(medium_tf.value, [])
                    )
                    combinations.append({
                        'timeframes': [short_tf.value, medium_tf.value],
                        'success_rate': combo_success,
                        'type': 'short_medium'
                    })
            
            # Store best combinations
            best_combinations = sorted(combinations, key=lambda x: x['success_rate'], reverse=True)[:10]
            self.analysis_results['best_combinations'] = best_combinations
            
        except Exception as e:
            logger.error(f"Error finding optimal entry combinations: {e}")
            
    def _analyze_timeframe_hierarchies(self, all_patterns: Dict[str, List[HistoricalPattern]]):
        """Analyze timeframe hierarchies for trend alignment"""
        try:
            # Analyze how higher timeframe trends affect lower timeframe patterns
            hierarchy_analysis = {}
            
            timeframe_hierarchy = [
                [Timeframe.MN1, Timeframe.W1, Timeframe.D1],  # Long-term
                [Timeframe.D1, Timeframe.H4, Timeframe.H1],   # Medium-term
                [Timeframe.H1, Timeframe.M15, Timeframe.M5]   # Short-term
            ]
            
            for hierarchy in timeframe_hierarchy:
                hierarchy_key = '_'.join([tf.value for tf in hierarchy])
                success_rates = []
                
                for tf in hierarchy:
                    patterns = all_patterns.get(tf.value, [])
                    if patterns:
                        avg_success = sum(p.success_rate for p in patterns) / len(patterns)
                        success_rates.append(avg_success)
                
                if success_rates:
                    hierarchy_analysis[hierarchy_key] = {
                        'avg_success_rate': sum(success_rates) / len(success_rates),
                        'timeframes': [tf.value for tf in hierarchy],
                        'pattern_count': sum(len(all_patterns.get(tf.value, [])) for tf in hierarchy)
                    }
            
            self.analysis_results['hierarchy_analysis'] = hierarchy_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe hierarchies: {e}")
            
    def _optimize_exit_strategies(self, all_patterns: Dict[str, List[HistoricalPattern]]):
        """Optimize exit strategies based on timeframe analysis"""
        try:
            # Analyze optimal exit strategies for different timeframes
            exit_optimization = {}
            
            for tf_name, patterns in all_patterns.items():
                if not patterns:
                    continue
                    
                # Calculate optimal profit targets and stop losses
                profit_targets = [p.avg_profit for p in patterns if p.avg_profit > 0]
                stop_losses = [p.avg_loss for p in patterns if p.avg_loss > 0]
                risk_rewards = [p.risk_reward_ratio for p in patterns if p.risk_reward_ratio > 0]
                
                if profit_targets and stop_losses and risk_rewards:
                    exit_optimization[tf_name] = {
                        'optimal_profit_target': np.percentile(profit_targets, 75),  # 75th percentile
                        'optimal_stop_loss': np.percentile(stop_losses, 25),        # 25th percentile
                        'avg_risk_reward': np.mean(risk_rewards),
                        'pattern_count': len(patterns)
                    }
            
            self.analysis_results['exit_optimization'] = exit_optimization
            
        except Exception as e:
            logger.error(f"Error optimizing exit strategies: {e}")
            
    def _calculate_combination_success(self, patterns1: List[HistoricalPattern], 
                                     patterns2: List[HistoricalPattern]) -> float:
        """Calculate success rate for timeframe combination"""
        try:
            if not patterns1 or not patterns2:
                return 0.0
                
            # Simple combination success calculation
            avg_success1 = sum(p.success_rate for p in patterns1) / len(patterns1)
            avg_success2 = sum(p.success_rate for p in patterns2) / len(patterns2)
            
            # Weighted combination (higher weight for alignment)
            combination_success = (avg_success1 * 0.4) + (avg_success2 * 0.6)
            
            return combination_success
            
        except Exception as e:
            logger.error(f"Error calculating combination success: {e}")
            return 0.0
            
    def get_best_patterns_for_style(self, trading_style: TradingStyle, limit: int = 10) -> List[HistoricalPattern]:
        """Get best patterns for specific trading style"""
        try:
            relevant_timeframes = list(self.timeframe_weights[trading_style].keys())
            all_patterns = []
            
            for tf in relevant_timeframes:
                patterns = self.pattern_db.get_patterns_by_timeframe(tf.value)
                all_patterns.extend(patterns)
            
            # Sort by success rate and risk/reward ratio
            sorted_patterns = sorted(
                all_patterns, 
                key=lambda p: p.success_rate * p.risk_reward_ratio, 
                reverse=True
            )
            
            return sorted_patterns[:limit]
            
        except Exception as e:
            logger.error(f"Error getting best patterns for style: {e}")
            return []
            
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        return {
            'learning_active': self.learning_active,
            'analysis_results': dict(self.analysis_results),
            'total_patterns_stored': len(self.pattern_db.get_patterns_by_timeframe('')),
            'last_update': datetime.now().isoformat()
        }
        
    def stop_learning(self):
        """Stop the continuous learning process"""
        self.learning_active = False
        logger.info("Stopped continuous multi-timeframe learning")

# Global multi-timeframe analyzer instance
mtf_analyzer = MultiTimeframeAnalyzer()

def start_multi_timeframe_learning():
    """Start the multi-timeframe learning system"""
    mtf_analyzer.start_continuous_learning()
    
def get_mtf_analysis_summary():
    """Get multi-timeframe analysis summary"""
    return mtf_analyzer.get_analysis_summary()

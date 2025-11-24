#!/usr/bin/env python3
"""
Paper Trading Simulator for BubbyBot Enhanced V2
Test strategies without risking real capital
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
import yfinance as yf
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTradingSimulator:
    """Simulate trading without real money"""
    
    def __init__(self, initial_balance: float = 10000.0, db_path: str = 'bubbybot_paper.db'):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.db_path = db_path
        self.open_positions = {}
        self.closed_trades = []
        
        # Initialize database
        self.init_database()
        
        logger.info(f"Paper Trading initialized with ${initial_balance:,.2f}")
    
    def init_database(self):
        """Initialize paper trading database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                leverage REAL DEFAULT 1.0,
                pnl REAL DEFAULT 0,
                pnl_percentage REAL DEFAULT 0,
                pattern_type TEXT,
                confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open',
                exit_timestamp TEXT,
                exit_reason TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_balance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                pnl REAL DEFAULT 0,
                trades_count INTEGER DEFAULT 0
            )
        ''')
        
        # Record initial balance
        cursor.execute('''
            INSERT INTO paper_balance (timestamp, balance, equity, pnl, trades_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), self.initial_balance, self.initial_balance, 0, 0))
        
        conn.commit()
        conn.close()
        
        logger.info("Paper trading database initialized")
    
    def open_position(self, signal: Dict) -> Dict:
        """Open a paper trading position"""
        try:
            # Calculate position size based on risk management
            risk_amount = self.current_balance * 0.02  # 2% risk per trade
            
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            leverage = signal.get('optimal_leverage', 5.0)
            
            # Calculate quantity based on risk
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance > 0:
                quantity = (risk_amount / stop_distance) * leverage
            else:
                quantity = (self.current_balance * 0.1) / entry_price  # 10% of balance
            
            # Create position
            position = {
                'symbol': signal['symbol'],
                'side': 'long' if signal['direction'] == 'bullish' else 'short',
                'entry_price': entry_price,
                'quantity': quantity,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': signal['take_profit'],
                'pattern': signal.get('pattern', 'unknown'),
                'confidence': signal.get('confidence', 70.0),
                'timestamp': datetime.now().isoformat(),
                'entry_value': entry_price * quantity
            }
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO paper_trades 
                (timestamp, symbol, side, entry_price, quantity, leverage, 
                 pattern_type, confidence, stop_loss, take_profit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position['timestamp'], position['symbol'], position['side'],
                position['entry_price'], position['quantity'], position['leverage'],
                position['pattern'], position['confidence'],
                position['stop_loss'], position['take_profit'], 'open'
            ))
            
            position_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Add to open positions
            self.open_positions[position_id] = position
            
            logger.info(f"Paper trade opened: {position['side']} {position['symbol']} @ ${entry_price:.4f} (Leverage: {leverage}x)")
            
            return {
                'status': 'success',
                'position_id': position_id,
                'position': position
            }
            
        except Exception as e:
            logger.error(f"Error opening paper position: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def close_position(self, position_id: int, exit_price: float, reason: str = 'manual') -> Dict:
        """Close a paper trading position"""
        try:
            if position_id not in self.open_positions:
                return {'status': 'error', 'message': 'Position not found'}
            
            position = self.open_positions[position_id]
            
            # Calculate P&L
            entry_price = position['entry_price']
            quantity = position['quantity']
            leverage = position['leverage']
            
            if position['side'] == 'long':
                price_change = exit_price - entry_price
            else:
                price_change = entry_price - exit_price
            
            pnl = (price_change / entry_price) * (entry_price * quantity) * leverage
            pnl_percentage = (price_change / entry_price) * 100 * leverage
            
            # Update balance
            self.current_balance += pnl
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE paper_trades 
                SET exit_price = ?, pnl = ?, pnl_percentage = ?, 
                    status = 'closed', exit_timestamp = ?, exit_reason = ?
                WHERE id = ?
            ''', (exit_price, pnl, pnl_percentage, datetime.now().isoformat(), reason, position_id))
            
            conn.commit()
            conn.close()
            
            # Move to closed trades
            position['exit_price'] = exit_price
            position['pnl'] = pnl
            position['pnl_percentage'] = pnl_percentage
            position['exit_reason'] = reason
            
            self.closed_trades.append(position)
            del self.open_positions[position_id]
            
            logger.info(f"Paper trade closed: {position['symbol']} - P&L: ${pnl:+.2f} ({pnl_percentage:+.2f}%) - Reason: {reason}")
            
            # Update balance record
            self.update_balance_record()
            
            return {
                'status': 'success',
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'new_balance': self.current_balance
            }
            
        except Exception as e:
            logger.error(f"Error closing paper position: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def check_stop_loss_take_profit(self):
        """Check all open positions for stop loss or take profit hits"""
        positions_to_close = []
        
        for position_id, position in self.open_positions.items():
            try:
                # Get current price
                symbol = position['symbol']
                ticker = yf.Ticker(symbol)
                current_data = ticker.history(period="1d", interval="1m")
                
                if current_data.empty:
                    continue
                
                current_price = current_data['Close'].iloc[-1]
                
                # Check stop loss and take profit
                should_close = False
                reason = ''
                
                if position['side'] == 'long':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        reason = 'stop_loss'
                    elif current_price >= position['take_profit']:
                        should_close = True
                        reason = 'take_profit'
                else:  # short
                    if current_price >= position['stop_loss']:
                        should_close = True
                        reason = 'stop_loss'
                    elif current_price <= position['take_profit']:
                        should_close = True
                        reason = 'take_profit'
                
                if should_close:
                    positions_to_close.append((position_id, current_price, reason))
                    
            except Exception as e:
                logger.error(f"Error checking position {position_id}: {e}")
        
        # Close positions that hit stop loss or take profit
        for position_id, exit_price, reason in positions_to_close:
            self.close_position(position_id, exit_price, reason)
    
    def update_balance_record(self):
        """Update balance record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate total equity (balance + open positions value)
            equity = self.current_balance
            for position in self.open_positions.values():
                equity += position['entry_value']
            
            total_pnl = self.current_balance - self.initial_balance
            trades_count = len(self.closed_trades)
            
            cursor.execute('''
                INSERT INTO paper_balance (timestamp, balance, equity, pnl, trades_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), self.current_balance, equity, total_pnl, trades_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating balance record: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            if not self.closed_trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'current_balance': self.current_balance,
                    'roi': 0.0
                }
            
            winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in self.closed_trades if t['pnl'] <= 0]
            
            total_pnl = sum(t['pnl'] for t in self.closed_trades)
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            return {
                'total_trades': len(self.closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(self.closed_trades)) * 100,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'current_balance': self.current_balance,
                'initial_balance': self.initial_balance,
                'roi': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
                'open_positions': len(self.open_positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {}
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM paper_trades 
                WHERE status = 'closed'
                ORDER BY exit_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            trades = []
            
            for row in cursor.fetchall():
                trade = dict(zip(columns, row))
                trades.append(trade)
            
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def reset(self):
        """Reset paper trading account"""
        self.current_balance = self.initial_balance
        self.open_positions = {}
        self.closed_trades = []
        
        # Clear database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM paper_trades')
        cursor.execute('DELETE FROM paper_balance')
        cursor.execute('''
            INSERT INTO paper_balance (timestamp, balance, equity, pnl, trades_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), self.initial_balance, self.initial_balance, 0, 0))
        conn.commit()
        conn.close()
        
        logger.info("Paper trading account reset")

# Example usage
if __name__ == "__main__":
    simulator = PaperTradingSimulator(initial_balance=10000.0)
    
    # Example signal
    test_signal = {
        'symbol': 'AVAX-USD',
        'direction': 'bullish',
        'entry_price': 14.82,
        'stop_loss': 14.50,
        'take_profit': 15.50,
        'confidence': 85.0,
        'optimal_leverage': 5.0,
        'pattern': 'green_dot'
    }
    
    # Open position
    result = simulator.open_position(test_signal)
    print(f"Position opened: {result}")
    
    # Get stats
    stats = simulator.get_performance_stats()
    print(f"Performance: {stats}")


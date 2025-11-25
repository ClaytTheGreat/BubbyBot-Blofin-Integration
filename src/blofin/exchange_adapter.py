"""
Blofin Exchange Adapter
Connects BubbyBot's trading logic with Blofin API
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal from BubbyBot analyzers"""
    instrument: str
    side: str  # 'buy' or 'sell'
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    pattern_type: str
    timestamp: datetime
    metadata: Dict = None  # Optional metadata


@dataclass
class Position:
    """Position information"""
    inst_id: str
    position_side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin: float
    leverage: int
    liquidation_price: float


class BlofinExchangeAdapter:
    """
    Adapter between BubbyBot and Blofin API
    Translates BubbyBot signals into Blofin API calls
    """
    
    def __init__(self, api_client, config: Dict):
        """
        Initialize exchange adapter
        
        Args:
            api_client: BlofinAPIClient instance
            config: Configuration dictionary
        """
        self.api_client = api_client
        self.config = config
        self.active_positions = {}
        
        logger.info("Initialized Blofin Exchange Adapter")
    
    def execute_signal(self, signal: TradingSignal) -> Dict:
        """
        Execute a trading signal
        
        Args:
            signal: TradingSignal object
            
        Returns:
            dict: Execution result
        """
        try:
            logger.info(f"Executing signal: {signal.instrument} {signal.side} "
                       f"(confidence: {signal.confidence:.2f})")
            
            # Get account balance
            balance_response = self.api_client.get_account_balance()
            if balance_response['code'] != '0':
                raise Exception(f"Failed to get balance: {balance_response['msg']}")
            
            balance_data = balance_response['data'][0]
            available_balance = float(balance_data['available'])
            
            logger.info(f"Available balance: {available_balance} USDT")
            
            # Calculate position size
            position_size = self.calculate_position_size(
                signal=signal,
                account_balance=available_balance
            )
            
            # Determine leverage based on confidence
            leverage = self._calculate_leverage(signal.confidence)
            
            # Set leverage
            logger.info(f"Setting leverage to {leverage}x for {signal.instrument}")
            self.api_client.set_leverage(
                inst_id=signal.instrument,
                leverage=leverage,
                margin_mode=self.config['trading']['default_margin_mode']
            )
            
            # Place order with TP/SL
            logger.info(f"Placing {signal.side} order: {position_size} contracts")
            logger.info(f"Entry: {signal.entry_price}, TP: {signal.take_profit}, SL: {signal.stop_loss}")
            
            order_response = self.api_client.place_order_with_tpsl(
                inst_id=signal.instrument,
                side=signal.side,
                order_type="market",  # Use market orders for immediate execution
                size=str(position_size),
                tp_price=signal.take_profit,
                sl_price=signal.stop_loss,
                margin_mode=self.config['trading']['default_margin_mode'],
                position_side=self.config['trading']['position_mode']
            )
            
            if order_response['code'] == '0':
                logger.info(f"✅ Order executed successfully: {order_response['data']}")
                return {
                    'success': True,
                    'order_id': order_response['data'][0]['orderId'],
                    'signal': signal,
                    'position_size': position_size,
                    'leverage': leverage
                }
            else:
                logger.error(f"❌ Order failed: {order_response['msg']}")
                return {
                    'success': False,
                    'error': order_response['msg'],
                    'signal': signal
                }
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {
                'success': False,
                'error': str(e),
                'signal': signal
            }
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            signal: Trading signal
            account_balance: Available account balance
            
        Returns:
            float: Position size in contracts
        """
        # Get risk parameters from config
        max_position_pct = self.config['risk_management']['max_position_size_pct']
        
        # Calculate maximum position value
        max_position_value = account_balance * max_position_pct
        
        # Calculate risk amount (distance from entry to stop loss)
        if signal.side == 'buy':
            risk_per_contract = signal.entry_price - signal.stop_loss
        else:  # sell
            risk_per_contract = signal.stop_loss - signal.entry_price
        
        # Ensure risk is positive
        if risk_per_contract <= 0:
            logger.warning("Invalid risk calculation, using minimum position size")
            return 0.1  # Minimum size
        
        # Calculate position size based on risk
        # Risk per contract as percentage of entry price
        risk_pct = risk_per_contract / signal.entry_price
        
        # Adjust position size based on confidence
        confidence_multiplier = 0.5 + (signal.confidence * 0.5)  # 0.5 to 1.0
        
        # Calculate position size
        position_size = (max_position_value / signal.entry_price) * confidence_multiplier
        
        # Round to appropriate precision (usually 0.1 or 1)
        position_size = round(position_size, 1)
        
        # Ensure minimum size
        if position_size < 0.1:
            position_size = 0.1
        
        logger.info(f"Calculated position size: {position_size} contracts "
                   f"(confidence multiplier: {confidence_multiplier:.2f})")
        
        return position_size
    
    def _calculate_leverage(self, confidence: float) -> int:
        """
        Calculate leverage based on signal confidence
        
        Args:
            confidence: Signal confidence (0-1)
            
        Returns:
            int: Leverage value
        """
        max_leverage = self.config['trading']['max_leverage']
        default_leverage = self.config['trading']['default_leverage']
        
        # Scale leverage based on confidence
        # Low confidence (0.7): 20x
        # Medium confidence (0.8): 35x
        # High confidence (0.9+): 50x
        if confidence >= 0.9:
            leverage = max_leverage
        elif confidence >= 0.8:
            leverage = int(max_leverage * 0.7)
        else:
            leverage = int(max_leverage * 0.4)
        
        return min(leverage, max_leverage)
    
    def monitor_positions(self) -> List[Position]:
        """
        Monitor all open positions
        
        Returns:
            list: List of Position objects
        """
        try:
            positions_response = self.api_client.get_positions()
            
            if positions_response['code'] != '0':
                logger.error(f"Failed to get positions: {positions_response['msg']}")
                return []
            
            positions = []
            for pos_data in positions_response['data']:
                if float(pos_data.get('total', 0)) != 0:  # Only active positions
                    position = Position(
                        inst_id=pos_data['instId'],
                        position_side=pos_data['positionSide'],
                        size=float(pos_data['total']),
                        entry_price=float(pos_data['avgPrice']),
                        current_price=float(pos_data['last']),
                        unrealized_pnl=float(pos_data['upl']),
                        margin=float(pos_data['margin']),
                        leverage=int(pos_data['leverage']),
                        liquidation_price=float(pos_data['liqPr'])
                    )
                    positions.append(position)
                    
                    # Log position status
                    logger.info(f"Position: {position.inst_id} | "
                              f"Size: {position.size} | "
                              f"Entry: {position.entry_price} | "
                              f"Current: {position.current_price} | "
                              f"PnL: {position.unrealized_pnl:.2f} USDT")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            return []
    
    def close_all_positions(self) -> Dict:
        """
        Close all open positions
        
        Returns:
            dict: Results of close operations
        """
        positions = self.monitor_positions()
        results = []
        
        for position in positions:
            try:
                logger.info(f"Closing position: {position.inst_id}")
                response = self.api_client.close_position(
                    inst_id=position.inst_id,
                    position_side=position.position_side,
                    margin_mode=self.config['trading']['default_margin_mode']
                )
                results.append({
                    'inst_id': position.inst_id,
                    'success': response['code'] == '0',
                    'response': response
                })
            except Exception as e:
                logger.error(f"Error closing position {position.inst_id}: {e}")
                results.append({
                    'inst_id': position.inst_id,
                    'success': False,
                    'error': str(e)
                })
        
        return {'results': results}
    
    def get_account_status(self) -> Dict:
        """
        Get comprehensive account status
        
        Returns:
            dict: Account status including balance, positions, and risk metrics
        """
        try:
            # Get balance
            balance_response = self.api_client.get_account_balance()
            balance_data = balance_response['data'][0]
            
            # Get positions
            positions = self.monitor_positions()
            
            # Calculate total exposure
            total_margin_used = sum(pos.margin for pos in positions)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'balance': {
                    'total_equity': float(balance_data['totalEquity']),
                    'available': float(balance_data['available']),
                    'margin_used': total_margin_used,
                    'unrealized_pnl': total_unrealized_pnl
                },
                'positions': {
                    'count': len(positions),
                    'details': [
                        {
                            'instrument': pos.inst_id,
                            'side': pos.position_side,
                            'size': pos.size,
                            'entry': pos.entry_price,
                            'current': pos.current_price,
                            'pnl': pos.unrealized_pnl,
                            'leverage': pos.leverage
                        }
                        for pos in positions
                    ]
                },
                'risk_metrics': {
                    'margin_usage_pct': (total_margin_used / float(balance_data['totalEquity'])) * 100 
                                       if float(balance_data['totalEquity']) > 0 else 0,
                    'pnl_pct': (total_unrealized_pnl / float(balance_data['totalEquity'])) * 100 
                              if float(balance_data['totalEquity']) > 0 else 0
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting account status: {e}")
            return {'error': str(e)}
    
    def check_risk_limits(self) -> Dict:
        """
        Check if current positions violate risk limits
        
        Returns:
            dict: Risk check results
        """
        status = self.get_account_status()
        
        if 'error' in status:
            return {'passed': False, 'error': status['error']}
        
        violations = []
        
        # Check daily loss limit
        max_daily_loss_pct = self.config['risk_management']['max_daily_loss_pct'] * 100
        current_pnl_pct = status['risk_metrics']['pnl_pct']
        
        if current_pnl_pct < -max_daily_loss_pct:
            violations.append(f"Daily loss limit exceeded: {current_pnl_pct:.2f}% < -{max_daily_loss_pct}%")
        
        # Check margin usage
        margin_usage = status['risk_metrics']['margin_usage_pct']
        if margin_usage > 80:  # Warning at 80% margin usage
            violations.append(f"High margin usage: {margin_usage:.2f}%")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'status': status
        }


if __name__ == "__main__":
    # Example usage
    from api_client import BlofinAPIClient
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Sample config
    config = {
        'trading': {
            'default_leverage': 50,
            'max_leverage': 50,
            'default_margin_mode': 'isolated',
            'position_mode': 'net'
        },
        'risk_management': {
            'max_position_size_pct': 0.05,
            'max_daily_loss_pct': 0.10,
            'default_stop_loss_pct': 0.02,
            'default_take_profit_pct': 0.06
        }
    }
    
    # Initialize
    client = BlofinAPIClient(
        api_key=os.getenv('BLOFIN_API_KEY', 'test'),
        secret_key=os.getenv('BLOFIN_SECRET_KEY', 'test'),
        passphrase=os.getenv('BLOFIN_PASSPHRASE', 'test'),
        demo=True
    )
    
    adapter = BlofinExchangeAdapter(client, config)
    
    # Get account status
    status = adapter.get_account_status()
    logger.info(f"Account status: {status}")

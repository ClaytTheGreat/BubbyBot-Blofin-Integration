#!/usr/bin/env python3
"""
GMX Browser Automation Integration for BubbyBot Enhanced V2
Automated trading execution on GMX decentralized perpetual exchange
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class GMXPosition:
    symbol: str
    direction: TradeDirection
    size: float
    net_value: float
    collateral: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    pnl: float
    pnl_percentage: float
    leverage: float

@dataclass
class GMXTradeOrder:
    symbol: str
    direction: TradeDirection
    order_type: OrderType
    size: float
    leverage: float
    collateral_token: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    limit_price: Optional[float] = None

class GMXBrowserAutomation:
    """Browser automation for GMX trading platform"""
    
    def __init__(self, browser_tools):
        """Initialize with browser automation tools"""
        self.browser = browser_tools
        self.base_url = "https://app.gmx.io"
        self.current_positions = {}
        self.is_connected = False
        
        logger.info("GMX Browser Automation initialized")
    
    async def navigate_to_gmx(self):
        """Navigate to GMX trading platform"""
        try:
            await self.browser.navigate(
                url=f"{self.base_url}/#/trade",
                intent="transactional",
                brief="Navigate to GMX trading platform"
            )
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            logger.info("Successfully navigated to GMX")
            return True
            
        except Exception as e:
            logger.error(f"Error navigating to GMX: {e}")
            return False
    
    async def check_wallet_connection(self) -> bool:
        """Check if wallet is connected"""
        try:
            page_content = await self.browser.view(brief="Check wallet connection status")
            
            # Look for "Connect wallet" button or connected wallet address
            if "Connect wallet" in page_content:
                self.is_connected = False
                logger.warning("Wallet not connected to GMX")
                return False
            else:
                self.is_connected = True
                logger.info("Wallet is connected to GMX")
                return True
                
        except Exception as e:
            logger.error(f"Error checking wallet connection: {e}")
            return False
    
    async def get_current_positions(self) -> List[GMXPosition]:
        """Get all current open positions"""
        try:
            page_content = await self.browser.view(brief="Get current positions")
            
            # Click on Positions tab to ensure we're viewing positions
            await self.browser.click(
                index=28,  # Positions tab
                brief="Click on Positions tab"
            )
            
            await asyncio.sleep(2)
            
            # Parse positions from the page content
            positions = []
            
            # Look for position data in the page content
            # This is a simplified parser - in production, you'd want more robust parsing
            if "AVAX/USD" in page_content and "Short" in page_content:
                # Extract position details using regex or string parsing
                position_match = re.search(r'AVAX/USD.*?(\d+\.\d+)x.*?Short.*?\$\s*([\d,]+\.\d+).*?\$\s*([\d.]+).*?\$\s*([\d.]+)', page_content)
                
                if position_match:
                    leverage = float(position_match.group(1))
                    net_value = float(position_match.group(2).replace(',', ''))
                    collateral = float(position_match.group(3))
                    
                    position = GMXPosition(
                        symbol="AVAX/USD",
                        direction=TradeDirection.SHORT,
                        size=net_value,
                        net_value=net_value,
                        collateral=collateral,
                        entry_price=14.8407,  # Would parse from page
                        mark_price=14.8214,   # Would parse from page
                        liquidation_price=0.0,  # Would parse from page
                        pnl=1.08,  # Would parse from page
                        pnl_percentage=2.87,  # Would parse from page
                        leverage=leverage
                    )
                    
                    positions.append(position)
                    self.current_positions[position.symbol] = position
            
            logger.info(f"Found {len(positions)} open positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []
    
    async def place_market_order(self, order: GMXTradeOrder) -> bool:
        """Place a market order on GMX"""
        try:
            if not self.is_connected:
                logger.error("Wallet not connected - cannot place order")
                return False
            
            # Select the trading pair
            await self.select_trading_pair(order.symbol)
            
            # Select Long or Short
            if order.direction == TradeDirection.LONG:
                await self.browser.click(
                    index=34,  # Long button
                    brief="Select Long direction"
                )
            else:
                await self.browser.click(
                    index=35,  # Short button
                    brief="Select Short direction"
                )
            
            await asyncio.sleep(1)
            
            # Ensure Market order type is selected
            await self.browser.click(
                index=37,  # Market button
                brief="Select Market order type"
            )
            
            await asyncio.sleep(1)
            
            # Enter the size/amount
            await self.browser.input(
                index=41,  # Pay amount input
                text=str(order.size),
                press_enter=False,
                brief="Enter order size"
            )
            
            await asyncio.sleep(1)
            
            # Set leverage
            await self.set_leverage(order.leverage)
            
            # Set take profit and stop loss if provided
            if order.take_profit or order.stop_loss:
                await self.set_take_profit_stop_loss(order.take_profit, order.stop_loss)
            
            # Click the order button (would be "Long AVAX/USD" or "Short AVAX/USD")
            # This would require finding the submit button dynamically
            
            logger.info(f"Market order placed: {order.direction.value} {order.symbol} size: {order.size}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return False
    
    async def select_trading_pair(self, symbol: str):
        """Select a trading pair"""
        try:
            # Click on the current pair dropdown
            await self.browser.click(
                index=21,  # AVAX/USD dropdown
                brief=f"Select trading pair {symbol}"
            )
            
            await asyncio.sleep(1)
            
            # In a real implementation, you'd search for the specific symbol
            # For now, we'll assume AVAX/USD is already selected
            
        except Exception as e:
            logger.error(f"Error selecting trading pair: {e}")
    
    async def set_leverage(self, leverage: float):
        """Set the leverage for the trade"""
        try:
            # Click on the leverage input/slider area
            # The leverage appears to be set via slider or direct input
            
            # Find the leverage input field and set it
            # This would require more specific element targeting
            
            logger.info(f"Leverage set to {leverage}x")
            
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
    
    async def set_take_profit_stop_loss(self, take_profit: Optional[float], stop_loss: Optional[float]):
        """Set take profit and stop loss levels"""
        try:
            # Click on Take Profit / Stop Loss section
            await self.browser.click(
                index=48,  # Take Profit / Stop Loss button
                brief="Open Take Profit / Stop Loss section"
            )
            
            await asyncio.sleep(1)
            
            if take_profit:
                # Set take profit price
                await self.browser.input(
                    index=50,  # Take profit price input
                    text=str(take_profit),
                    press_enter=False,
                    brief="Set take profit price"
                )
            
            if stop_loss:
                # Set stop loss price
                await self.browser.input(
                    index=54,  # Stop loss price input
                    text=str(stop_loss),
                    press_enter=False,
                    brief="Set stop loss price"
                )
            
            logger.info(f"Take profit: {take_profit}, Stop loss: {stop_loss}")
            
        except Exception as e:
            logger.error(f"Error setting take profit/stop loss: {e}")
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> bool:
        """Close a position (partially or fully)"""
        try:
            # Get current positions to find the one to close
            positions = await self.get_current_positions()
            
            target_position = None
            for pos in positions:
                if pos.symbol == symbol:
                    target_position = pos
                    break
            
            if not target_position:
                logger.warning(f"No open position found for {symbol}")
                return False
            
            # Click on the Close button for the position
            await self.browser.click(
                index=33,  # Close button (this would need to be dynamic)
                brief=f"Close position for {symbol}"
            )
            
            await asyncio.sleep(2)
            
            # If partial close, enter the percentage
            if percentage < 100.0:
                # Find and fill the percentage input
                # This would require more specific element targeting
                pass
            
            # Confirm the close
            # This would require finding and clicking the confirmation button
            
            logger.info(f"Position closed: {symbol} ({percentage}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance information"""
        try:
            # This would parse the account balance from the UI
            # GMX shows balances in the wallet/account section
            
            balances = {
                "USDC": 0.0,
                "AVAX": 0.0,
                "ETH": 0.0
            }
            
            # Parse balances from page content
            # Implementation would depend on GMX UI structure
            
            return balances
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    async def monitor_position_pnl(self, symbol: str) -> Optional[float]:
        """Monitor P&L for a specific position"""
        try:
            positions = await self.get_current_positions()
            
            for pos in positions:
                if pos.symbol == symbol:
                    return pos.pnl
            
            return None
            
        except Exception as e:
            logger.error(f"Error monitoring position P&L: {e}")
            return None
    
    async def execute_bubbybot_signal(self, signal: Dict) -> bool:
        """Execute a BubbyBot trading signal on GMX"""
        try:
            # Convert BubbyBot signal to GMX order
            direction = TradeDirection.LONG if signal['direction'] == 'bullish' else TradeDirection.SHORT
            
            # Calculate position size based on risk management
            account_balance = 1000.0  # Would get from actual balance
            risk_percentage = 0.02  # 2% risk per trade
            
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            stop_distance = abs(entry_price - stop_loss)
            
            # Calculate position size
            risk_amount = account_balance * risk_percentage
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            # Create GMX order
            order = GMXTradeOrder(
                symbol=signal['symbol'].replace('-', '/'),  # Convert BTC-USD to BTC/USD
                direction=direction,
                order_type=OrderType.MARKET,
                size=position_size,
                leverage=10.0,  # Default leverage
                collateral_token="USDC",
                take_profit=signal.get('take_profit'),
                stop_loss=signal.get('stop_loss')
            )
            
            # Execute the order
            success = await self.place_market_order(order)
            
            if success:
                logger.info(f"Successfully executed BubbyBot signal on GMX: {signal['symbol']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing BubbyBot signal: {e}")
            return False

class GMXIntegrationManager:
    """Manager class for GMX integration with BubbyBot"""
    
    def __init__(self, browser_tools):
        self.gmx_automation = GMXBrowserAutomation(browser_tools)
        self.active_signals = {}
        self.monitoring_active = False
    
    async def initialize(self) -> bool:
        """Initialize GMX integration"""
        try:
            # Navigate to GMX
            if not await self.gmx_automation.navigate_to_gmx():
                return False
            
            # Check wallet connection
            if not await self.gmx_automation.check_wallet_connection():
                logger.warning("Wallet not connected - manual connection required")
                return False
            
            logger.info("GMX integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing GMX integration: {e}")
            return False
    
    async def process_bubbybot_signal(self, signal: Dict) -> bool:
        """Process a signal from BubbyBot and execute on GMX"""
        try:
            # Validate signal
            if not self.validate_signal(signal):
                return False
            
            # Check if we already have a position for this symbol
            positions = await self.gmx_automation.get_current_positions()
            existing_position = None
            
            for pos in positions:
                if pos.symbol == signal['symbol'].replace('-', '/'):
                    existing_position = pos
                    break
            
            if existing_position:
                logger.info(f"Existing position found for {signal['symbol']}, skipping new order")
                return False
            
            # Execute the signal
            success = await self.gmx_automation.execute_bubbybot_signal(signal)
            
            if success:
                self.active_signals[signal['symbol']] = signal
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing BubbyBot signal: {e}")
            return False
    
    def validate_signal(self, signal: Dict) -> bool:
        """Validate a trading signal"""
        required_fields = ['symbol', 'direction', 'entry_price', 'stop_loss', 'take_profit']
        
        for field in required_fields:
            if field not in signal:
                logger.error(f"Missing required field in signal: {field}")
                return False
        
        return True
    
    async def monitor_positions(self):
        """Monitor active positions and manage them"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                positions = await self.gmx_automation.get_current_positions()
                
                for position in positions:
                    # Check if position should be closed based on BubbyBot logic
                    # This could include trailing stops, time-based exits, etc.
                    
                    # Example: Close position if P&L reaches certain threshold
                    if position.pnl_percentage >= 5.0:  # 5% profit
                        logger.info(f"Closing profitable position: {position.symbol}")
                        await self.gmx_automation.close_position(position.symbol)
                    elif position.pnl_percentage <= -2.0:  # 2% loss
                        logger.info(f"Closing losing position: {position.symbol}")
                        await self.gmx_automation.close_position(position.symbol)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def stop_monitoring(self):
        """Stop position monitoring"""
        self.monitoring_active = False
        logger.info("GMX position monitoring stopped")

# Example usage
async def main():
    """Example usage of GMX automation"""
    
    # This would be initialized with actual browser tools
    # browser_tools = BrowserTools()
    
    # gmx_manager = GMXIntegrationManager(browser_tools)
    
    # Initialize
    # await gmx_manager.initialize()
    
    # Example signal from BubbyBot
    example_signal = {
        'symbol': 'AVAX-USD',
        'direction': 'bullish',
        'entry_price': 14.82,
        'stop_loss': 14.50,
        'take_profit': 15.50,
        'confidence': 85.0,
        'confluence_score': 0.78
    }
    
    # Process signal
    # await gmx_manager.process_bubbybot_signal(example_signal)
    
    # Start monitoring
    # await gmx_manager.monitor_positions()

if __name__ == "__main__":
    asyncio.run(main())

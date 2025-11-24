"""
Blofin API Client
Handles all REST API interactions with Blofin exchange
"""

import hmac
import hashlib
import base64
import json
import requests
import time
from datetime import datetime
from uuid import uuid4
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlofinAPIClient:
    """
    Main API client for Blofin exchange
    Handles authentication, request signing, and all API endpoints
    """
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str, demo: bool = False):
        """
        Initialize Blofin API client
        
        Args:
            api_key: API key from Blofin
            secret_key: Secret key from Blofin
            passphrase: Passphrase set during API key creation
            demo: If True, use demo trading environment
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.demo = demo
        
        # Set base URL based on environment
        if demo:
            self.base_url = "https://demo-trading-openapi.blofin.com"
        else:
            self.base_url = "https://openapi.blofin.com"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'ACCESS-KEY': self.api_key,
            'ACCESS-PASSPHRASE': self.passphrase
        })
        
        logger.info(f"Initialized Blofin API client (demo={demo})")
    
    def _generate_signature(self, method: str, path: str, body: Optional[Dict] = None) -> tuple:
        """
        Generate HMAC-SHA256 signature for API request
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path (including query params)
            body: Request body for POST/PUT requests
            
        Returns:
            tuple: (signature, timestamp, nonce)
        """
        timestamp = str(int(datetime.now().timestamp() * 1000))
        nonce = str(uuid4())
        
        # Create prehash string
        prehash = f"{path}{method}{timestamp}{nonce}"
        if body:
            prehash += json.dumps(body, separators=(',', ':'))
        
        # Generate hex signature and convert to base64
        hex_signature = hmac.new(
            self.secret_key.encode(),
            prehash.encode(),
            hashlib.sha256
        ).hexdigest().encode()
        
        signature = base64.b64encode(hex_signature).decode()
        
        return signature, timestamp, nonce
    
    def _make_request(self, method: str, path: str, params: Optional[Dict] = None, 
                     body: Optional[Dict] = None) -> Dict:
        """
        Make authenticated API request
        
        Args:
            method: HTTP method
            path: API endpoint path
            params: Query parameters for GET requests
            body: Request body for POST requests
            
        Returns:
            dict: API response
        """
        # Build full URL
        url = f"{self.base_url}{path}"
        
        # Add query params to path for signature
        if params and method == 'GET':
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature_path = f"{path}?{query_string}"
        else:
            signature_path = path
        
        # Generate signature
        signature, timestamp, nonce = self._generate_signature(method, signature_path, body)
        
        # Update headers
        headers = {
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-NONCE': nonce
        }
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = self.session.post(url, json=body, headers=headers)
            elif method == 'DELETE':
                response = self.session.delete(url, json=body, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            # Check for API errors
            if result.get('code') != '0' and result.get('code') != 0:
                logger.error(f"API Error: {result}")
                raise Exception(f"API Error: {result.get('msg', 'Unknown error')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    # ==================== Account Endpoints ====================
    
    def get_account_balance(self, product_type: str = "USDT-FUTURES") -> Dict:
        """
        Get account balance
        
        Args:
            product_type: Product type (USDT-FUTURES, COIN-FUTURES, etc.)
            
        Returns:
            dict: Account balance information
        """
        path = "/api/v1/account/balance"
        params = {"productType": product_type}
        return self._make_request('GET', path, params=params)
    
    def get_positions(self, inst_id: Optional[str] = None) -> Dict:
        """
        Get current positions
        
        Args:
            inst_id: Instrument ID (e.g., BTC-USDT). If None, returns all positions
            
        Returns:
            dict: Position information
        """
        path = "/api/v1/account/positions"
        params = {}
        if inst_id:
            params['instId'] = inst_id
        return self._make_request('GET', path, params=params)
    
    def set_leverage(self, inst_id: str, leverage: int, margin_mode: str) -> Dict:
        """
        Set leverage for an instrument
        
        Args:
            inst_id: Instrument ID (e.g., BTC-USDT)
            leverage: Leverage value (1-150)
            margin_mode: Margin mode (cross or isolated)
            
        Returns:
            dict: API response
        """
        path = "/api/v1/account/set-leverage"
        body = {
            "instId": inst_id,
            "leverage": str(leverage),
            "marginMode": margin_mode
        }
        return self._make_request('POST', path, body=body)
    
    def set_margin_mode(self, inst_id: str, margin_mode: str) -> Dict:
        """
        Set margin mode for an instrument
        
        Args:
            inst_id: Instrument ID
            margin_mode: Margin mode (cross or isolated)
            
        Returns:
            dict: API response
        """
        path = "/api/v1/account/set-margin-mode"
        body = {
            "instId": inst_id,
            "marginMode": margin_mode
        }
        return self._make_request('POST', path, body=body)
    
    # ==================== Trading Endpoints ====================
    
    def place_order(self, inst_id: str, side: str, order_type: str, size: str,
                   margin_mode: str = "isolated", position_side: str = "net",
                   price: Optional[str] = None, reduce_only: bool = False,
                   client_order_id: Optional[str] = None,
                   tp_trigger_price: Optional[str] = None, tp_order_price: Optional[str] = None,
                   sl_trigger_price: Optional[str] = None, sl_order_price: Optional[str] = None) -> Dict:
        """
        Place an order
        
        Args:
            inst_id: Instrument ID (e.g., BTC-USDT)
            side: Order side (buy or sell)
            order_type: Order type (market, limit, post_only, fok, ioc)
            size: Order size (number of contracts)
            margin_mode: Margin mode (cross or isolated)
            position_side: Position side (net, long, or short)
            price: Order price (required for limit orders)
            reduce_only: Whether order can only reduce position
            client_order_id: Client-assigned order ID
            tp_trigger_price: Take profit trigger price
            tp_order_price: Take profit order price (-1 for market)
            sl_trigger_price: Stop loss trigger price
            sl_order_price: Stop loss order price (-1 for market)
            
        Returns:
            dict: Order response with order ID
        """
        path = "/api/v1/trade/order"
        
        body = {
            "instId": inst_id,
            "marginMode": margin_mode,
            "positionSide": position_side,
            "side": side,
            "orderType": order_type,
            "size": size
        }
        
        # Add optional parameters
        if price:
            body['price'] = price
        if reduce_only:
            body['reduceOnly'] = "true"
        if client_order_id:
            body['clientOrderId'] = client_order_id
        
        # Add TP/SL if provided
        if tp_trigger_price and tp_order_price:
            body['tpTriggerPrice'] = tp_trigger_price
            body['tpOrderPrice'] = tp_order_price
        if sl_trigger_price and sl_order_price:
            body['slTriggerPrice'] = sl_trigger_price
            body['slOrderPrice'] = sl_order_price
        
        logger.info(f"Placing order: {inst_id} {side} {size} @ {price or 'market'}")
        return self._make_request('POST', path, body=body)
    
    def place_order_with_tpsl(self, inst_id: str, side: str, order_type: str, size: str,
                              tp_price: float, sl_price: float,
                              margin_mode: str = "isolated", position_side: str = "net",
                              price: Optional[str] = None) -> Dict:
        """
        Place an order with automatic TP/SL
        
        Args:
            inst_id: Instrument ID
            side: Order side (buy or sell)
            order_type: Order type
            size: Order size
            tp_price: Take profit price
            sl_price: Stop loss price
            margin_mode: Margin mode
            position_side: Position side
            price: Order price (for limit orders)
            
        Returns:
            dict: Order response
        """
        return self.place_order(
            inst_id=inst_id,
            side=side,
            order_type=order_type,
            size=size,
            margin_mode=margin_mode,
            position_side=position_side,
            price=price,
            tp_trigger_price=str(tp_price),
            tp_order_price=str(tp_price),
            sl_trigger_price=str(sl_price),
            sl_order_price="-1"  # Market price for SL
        )
    
    def cancel_order(self, inst_id: str, order_id: str) -> Dict:
        """
        Cancel an order
        
        Args:
            inst_id: Instrument ID
            order_id: Order ID to cancel
            
        Returns:
            dict: Cancellation response
        """
        path = "/api/v1/trade/cancel-order"
        body = {
            "instId": inst_id,
            "orderId": order_id
        }
        logger.info(f"Cancelling order: {order_id}")
        return self._make_request('POST', path, body=body)
    
    def cancel_multiple_orders(self, orders: List[Dict]) -> Dict:
        """
        Cancel multiple orders
        
        Args:
            orders: List of orders to cancel [{"instId": "BTC-USDT", "orderId": "123"}]
            
        Returns:
            dict: Cancellation response
        """
        path = "/api/v1/trade/cancel-batch-orders"
        body = orders
        return self._make_request('POST', path, body=body)
    
    def close_position(self, inst_id: str, position_side: str = "net", 
                      margin_mode: str = "isolated") -> Dict:
        """
        Close a position
        
        Args:
            inst_id: Instrument ID
            position_side: Position side (net, long, or short)
            margin_mode: Margin mode
            
        Returns:
            dict: Close position response
        """
        path = "/api/v1/trade/close-positions"
        body = {
            "instId": inst_id,
            "positionSide": position_side,
            "marginMode": margin_mode
        }
        logger.info(f"Closing position: {inst_id}")
        return self._make_request('POST', path, body=body)
    
    def get_active_orders(self, inst_id: Optional[str] = None) -> Dict:
        """
        Get active (pending) orders
        
        Args:
            inst_id: Instrument ID (optional)
            
        Returns:
            dict: Active orders
        """
        path = "/api/v1/trade/orders-pending"
        params = {}
        if inst_id:
            params['instId'] = inst_id
        return self._make_request('GET', path, params=params)
    
    def get_order_detail(self, inst_id: str, order_id: str) -> Dict:
        """
        Get order details
        
        Args:
            inst_id: Instrument ID
            order_id: Order ID
            
        Returns:
            dict: Order details
        """
        path = "/api/v1/trade/order"
        params = {
            "instId": inst_id,
            "orderId": order_id
        }
        return self._make_request('GET', path, params=params)
    
    def get_order_history(self, inst_id: Optional[str] = None, limit: int = 100) -> Dict:
        """
        Get order history
        
        Args:
            inst_id: Instrument ID (optional)
            limit: Number of records to return
            
        Returns:
            dict: Order history
        """
        path = "/api/v1/trade/orders-history"
        params = {"limit": str(limit)}
        if inst_id:
            params['instId'] = inst_id
        return self._make_request('GET', path, params=params)
    
    # ==================== Market Data Endpoints ====================
    
    def get_ticker(self, inst_id: str) -> Dict:
        """
        Get ticker data
        
        Args:
            inst_id: Instrument ID
            
        Returns:
            dict: Ticker data
        """
        path = "/api/v1/market/ticker"
        params = {"instId": inst_id}
        return self._make_request('GET', path, params=params)
    
    def get_tickers(self) -> Dict:
        """
        Get all tickers
        
        Returns:
            dict: All ticker data
        """
        path = "/api/v1/market/tickers"
        return self._make_request('GET', path)
    
    def get_orderbook(self, inst_id: str, depth: int = 20) -> Dict:
        """
        Get orderbook data
        
        Args:
            inst_id: Instrument ID
            depth: Orderbook depth (default 20)
            
        Returns:
            dict: Orderbook data
        """
        path = "/api/v1/market/books"
        params = {
            "instId": inst_id,
            "sz": str(depth)
        }
        return self._make_request('GET', path, params=params)
    
    def get_instruments(self, inst_type: str = "SWAP") -> Dict:
        """
        Get available instruments
        
        Args:
            inst_type: Instrument type (SWAP for perpetual contracts)
            
        Returns:
            dict: List of instruments
        """
        path = "/api/v1/market/instruments"
        params = {"instType": inst_type}
        return self._make_request('GET', path, params=params)
    
    def get_candles(self, inst_id: str, bar: str = "1m", limit: int = 100) -> Dict:
        """
        Get candlestick data
        
        Args:
            inst_id: Instrument ID
            bar: Bar size (1m, 5m, 15m, 30m, 1H, 4H, 1D, etc.)
            limit: Number of candles to return
            
        Returns:
            dict: Candlestick data
        """
        path = "/api/v1/market/candles"
        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(limit)
        }
        return self._make_request('GET', path, params=params)


if __name__ == "__main__":
    # Example usage (for testing only)
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize client (demo mode)
    client = BlofinAPIClient(
        api_key=os.getenv('BLOFIN_API_KEY', 'test'),
        secret_key=os.getenv('BLOFIN_SECRET_KEY', 'test'),
        passphrase=os.getenv('BLOFIN_PASSPHRASE', 'test'),
        demo=True
    )
    
    try:
        # Test getting account balance
        balance = client.get_account_balance()
        logger.info(f"Account balance: {balance}")
        
        # Test getting ticker
        ticker = client.get_ticker("BTC-USDT")
        logger.info(f"BTC-USDT ticker: {ticker}")
        
    except Exception as e:
        logger.error(f"Error: {e}")

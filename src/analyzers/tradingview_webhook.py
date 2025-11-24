#!/usr/bin/env python3
"""
TradingView Webhook Receiver for BubbyBot Enhanced V2
Receives real-time signals from TradingView alerts
"""

from flask import Flask, request, jsonify
import json
import logging
import hmac
import hashlib
from datetime import datetime
from typing import Dict, Optional
import asyncio
from threading import Thread
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingViewWebhook:
    """Webhook receiver for TradingView alerts"""
    
    def __init__(self, secret_key: str = "bubbybot_secret_2025"):
        self.secret_key = secret_key
        self.signal_queue = []
        self.db_path = 'bubbybot_gmx.db'
        
        logger.info("TradingView Webhook initialized")
    
    def verify_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature for security"""
        try:
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def parse_tradingview_alert(self, data: Dict) -> Optional[Dict]:
        """Parse TradingView alert into BubbyBot signal format"""
        try:
            # TradingView alert format:
            # {
            #   "ticker": "AVAXUSD",
            #   "action": "buy" or "sell",
            #   "price": 14.82,
            #   "indicator": "market_cipher" or "lux_algo" or "frankie_candles",
            #   "pattern": "green_dot" or "blood_diamond" etc,
            #   "timeframe": "15m",
            #   "timestamp": "2025-11-18T14:30:00Z"
            # }
            
            # Convert to BubbyBot signal format
            signal = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'symbol': self.normalize_symbol(data.get('ticker', '')),
                'direction': 'bullish' if data.get('action') == 'buy' else 'bearish',
                'indicator': data.get('indicator', 'unknown'),
                'pattern': data.get('pattern', 'unknown'),
                'entry_price': float(data.get('price', 0)),
                'timeframe': data.get('timeframe', '15m'),
                'source': 'tradingview',
                'raw_data': data
            }
            
            # Calculate stop loss and take profit based on pattern
            signal = self.calculate_risk_levels(signal)
            
            # Calculate confidence score
            signal['confidence'] = self.calculate_confidence(signal)
            
            logger.info(f"Parsed TradingView signal: {signal['symbol']} {signal['direction']} - {signal['pattern']}")
            return signal
            
        except Exception as e:
            logger.error(f"Error parsing TradingView alert: {e}")
            return None
    
    def normalize_symbol(self, ticker: str) -> str:
        """Normalize ticker symbol to BubbyBot format"""
        # Convert AVAXUSD to AVAX-USD, BTCUSD to BTC-USD, etc.
        ticker = ticker.upper().replace('USDT', '').replace('PERP', '')
        
        if 'USD' in ticker and not '-' in ticker:
            base = ticker.replace('USD', '')
            return f"{base}-USD"
        
        return ticker
    
    def calculate_risk_levels(self, signal: Dict) -> Dict:
        """Calculate stop loss and take profit levels"""
        entry_price = signal['entry_price']
        pattern = signal['pattern']
        direction = signal['direction']
        
        # Pattern-specific risk/reward ratios
        pattern_params = {
            'green_dot': {'stop_pct': 0.02, 'target_pct': 0.06, 'rr': 3.0},
            'blood_diamond': {'stop_pct': 0.02, 'target_pct': 0.05, 'rr': 2.5},
            'anchor_trigger': {'stop_pct': 0.015, 'target_pct': 0.06, 'rr': 4.0},
            'money_flow_div': {'stop_pct': 0.025, 'target_pct': 0.05, 'rr': 2.0},
            'squeeze_release': {'stop_pct': 0.02, 'target_pct': 0.07, 'rr': 3.5},
            'triple_confluence': {'stop_pct': 0.015, 'target_pct': 0.075, 'rr': 5.0},
            'default': {'stop_pct': 0.02, 'target_pct': 0.06, 'rr': 3.0}
        }
        
        params = pattern_params.get(pattern, pattern_params['default'])
        
        if direction == 'bullish':
            signal['stop_loss'] = entry_price * (1 - params['stop_pct'])
            signal['take_profit'] = entry_price * (1 + params['target_pct'])
        else:
            signal['stop_loss'] = entry_price * (1 + params['stop_pct'])
            signal['take_profit'] = entry_price * (1 - params['target_pct'])
        
        signal['risk_reward_ratio'] = params['rr']
        
        return signal
    
    def calculate_confidence(self, signal: Dict) -> float:
        """Calculate confidence score for the signal"""
        base_confidence = 70.0  # Base confidence for TradingView signals
        
        # Boost confidence for high-success patterns
        pattern_boost = {
            'green_dot': 19.0,           # 89% success
            'triple_confluence': 17.0,   # 87% success
            'blood_diamond': 14.0,       # 84% success
            'anchor_trigger': 12.0,      # 82% success
            'squeeze_release': 11.0,     # 81% success
            'default': 10.0
        }
        
        confidence = base_confidence + pattern_boost.get(signal['pattern'], pattern_boost['default'])
        
        # Boost for specific indicators
        if signal['indicator'] == 'market_cipher':
            confidence += 3.0
        elif signal['indicator'] == 'lux_algo':
            confidence += 2.0
        
        # Cap at 95%
        return min(confidence, 95.0)
    
    def store_signal(self, signal: Dict):
        """Store signal in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, signal_type, pattern_type, 
                                   confidence, entry_price, stop_loss, take_profit, 
                                   timeframe, platform)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'],
                signal['symbol'],
                signal['direction'],
                signal['pattern'],
                signal['confidence'],
                signal['entry_price'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['timeframe'],
                'tradingview'
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Signal stored in database: {signal['symbol']}")
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    def process_webhook(self, data: Dict, signature: Optional[str] = None) -> Dict:
        """Process incoming webhook"""
        try:
            # Verify signature if provided
            if signature:
                payload = json.dumps(data, sort_keys=True)
                if not self.verify_signature(payload, signature):
                    logger.warning("Invalid webhook signature")
                    return {'status': 'error', 'message': 'Invalid signature'}
            
            # Parse the alert
            signal = self.parse_tradingview_alert(data)
            
            if not signal:
                return {'status': 'error', 'message': 'Failed to parse alert'}
            
            # Store signal
            self.store_signal(signal)
            
            # Add to processing queue
            self.signal_queue.append(signal)
            
            return {
                'status': 'success',
                'message': 'Signal received and queued',
                'signal': {
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'pattern': signal['pattern'],
                    'confidence': signal['confidence']
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_pending_signals(self) -> list:
        """Get all pending signals from queue"""
        signals = self.signal_queue.copy()
        self.signal_queue.clear()
        return signals

# Flask app for webhook endpoint
app = Flask(__name__)
webhook_handler = TradingViewWebhook()

@app.route('/webhook/tradingview', methods=['POST'])
def tradingview_webhook():
    """Endpoint for TradingView webhooks"""
    try:
        # Get signature from header if present
        signature = request.headers.get('X-Webhook-Signature')
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Process webhook
        result = webhook_handler.process_webhook(data, signature)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Webhook endpoint error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/webhook/test', methods=['GET', 'POST'])
def test_webhook():
    """Test endpoint to verify webhook is working"""
    if request.method == 'POST':
        data = request.get_json()
        return jsonify({
            'status': 'success',
            'message': 'Webhook is working',
            'received': data
        }), 200
    else:
        return jsonify({
            'status': 'success',
            'message': 'Webhook endpoint is active',
            'endpoints': {
                'tradingview': '/webhook/tradingview',
                'test': '/webhook/test'
            }
        }), 200

@app.route('/signals/pending', methods=['GET'])
def get_pending_signals():
    """Get pending signals"""
    signals = webhook_handler.get_pending_signals()
    return jsonify({
        'status': 'success',
        'count': len(signals),
        'signals': signals
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'TradingView Webhook Receiver',
        'timestamp': datetime.now().isoformat()
    }), 200

def run_webhook_server(host='0.0.0.0', port=5001):
    """Run the webhook server"""
    logger.info(f"Starting TradingView webhook server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    run_webhook_server()


'''
Market Cipher & Lux Algo Trading Bot - Web Dashboard
'''

import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'market-cipher-lux-algo-secret-key')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Global variables for bot state
bot_status = {
    'running': True,
    'paper_trading': os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true',
    'last_signal': None,
    'total_trades': 0,
    'successful_trades': 0,
    'current_balance': 10000.0,  # Starting balance for paper trading
    'start_time': datetime.now().isoformat()
}

# Store recent signals
recent_signals = []

# Simple HTML template for dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Cipher & Lux Algo Trading Bot</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: white; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00ff88; font-size: 2.5rem; margin-bottom: 10px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px; border: 1px solid #00ff88; }
        .status-card h3 { color: #00ff88; margin-bottom: 10px; }
        .status-value { font-size: 1.8rem; font-weight: bold; margin-bottom: 5px; }
        .running { color: #00ff88; }
        .paper-trading { color: #ffaa00; }
        .signals-section { background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px; border: 1px solid #00ff88; }
        .signal-item { background: rgba(255,255,255,0.05); border-radius: 5px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #00ff88; }
        .no-signals { text-align: center; opacity: 0.7; padding: 40px; }
        .refresh-btn { background: #00ff88; color: #000; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer; margin-top: 20px; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
        .pulse { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Market Cipher & Lux Algo Trading Bot</h1>
            <p>AI-Powered Automated Trading with Continuous Learning</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>Bot Status</h3>
                <div class="status-value running pulse">🟢 RUNNING</div>
                <div>System Status</div>
            </div>
            
            <div class="status-card">
                <h3>Trading Mode</h3>
                <div class="status-value paper-trading">📝 PAPER TRADING</div>
                <div>Current Mode</div>
            </div>
            
            <div class="status-card">
                <h3>Total Trades</h3>
                <div class="status-value">{{ total_trades }}</div>
                <div>Executed Trades</div>
            </div>
            
            <div class="status-card">
                <h3>Success Rate</h3>
                <div class="status-value">{{ win_rate }}%</div>
                <div>Win Rate</div>
            </div>
            
            <div class="status-card">
                <h3>Current Balance</h3>
                <div class="status-value">${{ balance }}</div>
                <div>P&L: ${{ pnl }}</div>
            </div>
            
            <div class="status-card">
                <h3>Webhook URL</h3>
                <div style="font-size: 0.9rem; word-break: break-all; color: #00ff88;">{{ webhook_url }}/webhook/tradingview</div>
                <div>TradingView Integration</div>
            </div>
        </div>
        
        <div class="signals-section">
            <h2>📊 Recent Trading Signals</h2>
            {% if signals %}
                {% for signal in signals %}
                <div class="signal-item">
                    <strong>{{ signal.symbol }}</strong> - {{ signal.action.upper() }} - 
                    Confidence: {{ (signal.confidence * 100)|round(1) }}% - 
                    Time: {{ signal.timestamp }}
                </div>
                {% endfor %}
            {% else %}
                <div class="no-signals">
                    <h3>🔍 Waiting for Trading Signals</h3>
                    <p>The bot is monitoring Market Cipher and Lux Algo indicators...</p>
                    <p>Configure your TradingView alerts to send webhooks to:</p>
                    <p style="color: #00ff88; font-weight: bold;">{{ webhook_url }}/webhook/tradingview</p>
                </div>
            {% endif %}
            
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Dashboard</button>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() { location.reload(); }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main dashboard"""
    win_rate = (bot_status['successful_trades'] / max(bot_status['total_trades'], 1)) * 100
    pnl = bot_status['current_balance'] - 10000
    
    # Get the current URL for webhook display
    webhook_url = request.host_url.rstrip('/')
    
    return render_template_string(DASHBOARD_TEMPLATE,
                                total_trades=bot_status['total_trades'],
                                win_rate=round(win_rate, 1),
                                balance=round(bot_status['current_balance'], 2),
                                pnl=round(pnl, 2),
                                signals=recent_signals[-10:],
                                webhook_url=webhook_url)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_running': bot_status['running'],
        'paper_trading': bot_status['paper_trading'],
        'version': '1.0.0'
    })

@app.route('/api/status')
def api_status():
    """API endpoint for bot status"""
    return jsonify(bot_status)

@app.route('/api/signals')
def api_signals():
    """API endpoint for recent signals"""
    return jsonify({
        'signals': recent_signals[-20:],  # Last 20 signals
        'total_signals': len(recent_signals)
    })


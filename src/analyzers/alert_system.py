#!/usr/bin/env python3
"""
Alert System for BubbyBot Enhanced V2
Send notifications for trades, signals, and system events
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
import json
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertSystem:
    """Multi-channel alert system"""
    
    def __init__(self, config_path: str = "config/alert_config.json"):
        self.config = self.load_config(config_path)
        self.enabled_channels = []
        
        # Check which channels are configured
        if self.config.get('email', {}).get('enabled'):
            self.enabled_channels.append('email')
        if self.config.get('telegram', {}).get('enabled'):
            self.enabled_channels.append('telegram')
        if self.config.get('discord', {}).get('enabled'):
            self.enabled_channels.append('discord')
        
        logger.info(f"Alert system initialized with channels: {', '.join(self.enabled_channels)}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load alert configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default alert configuration"""
        return {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "recipient_email": ""
            },
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            },
            "discord": {
                "enabled": False,
                "webhook_url": ""
            },
            "alert_types": {
                "trade_opened": True,
                "trade_closed": True,
                "stop_loss_hit": True,
                "take_profit_hit": True,
                "high_confidence_signal": True,
                "daily_summary": True,
                "system_error": True
            }
        }
    
    def send_alert(self, alert_type: str, title: str, message: str, data: Optional[Dict] = None):
        """Send alert through all enabled channels"""
        if not self.config.get('alert_types', {}).get(alert_type, True):
            return  # Alert type is disabled
        
        logger.info(f"Sending alert: {alert_type} - {title}")
        
        for channel in self.enabled_channels:
            try:
                if channel == 'email':
                    self.send_email_alert(title, message, data)
                elif channel == 'telegram':
                    self.send_telegram_alert(title, message, data)
                elif channel == 'discord':
                    self.send_discord_alert(title, message, data)
            except Exception as e:
                logger.error(f"Error sending {channel} alert: {e}")
    
    def send_email_alert(self, title: str, message: str, data: Optional[Dict] = None):
        """Send email alert"""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = email_config['recipient_email']
            msg['Subject'] = f"BubbyBot Alert: {title}"
            
            # Create HTML email body
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2 style="color: #667eea;">{title}</h2>
                    <p>{message}</p>
                    {self.format_data_html(data) if data else ''}
                    <hr>
                    <p style="color: #666; font-size: 12px;">
                        BubbyBot Enhanced V2 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['sender_email'], email_config['sender_password'])
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def send_telegram_alert(self, title: str, message: str, data: Optional[Dict] = None):
        """Send Telegram alert"""
        try:
            telegram_config = self.config['telegram']
            
            # Format message
            text = f"🤖 *{title}*\n\n{message}"
            
            if data:
                text += "\n\n" + self.format_data_telegram(data)
            
            text += f"\n\n_BubbyBot Enhanced V2 | {datetime.now().strftime('%H:%M:%S')}_"
            
            # Send via Telegram Bot API
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
            payload = {
                'chat_id': telegram_config['chat_id'],
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            logger.info("Telegram alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def send_discord_alert(self, title: str, message: str, data: Optional[Dict] = None):
        """Send Discord alert"""
        try:
            discord_config = self.config['discord']
            
            # Create Discord embed
            embed = {
                "title": f"🤖 {title}",
                "description": message,
                "color": 6737130,  # Purple color
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "BubbyBot Enhanced V2"
                }
            }
            
            if data:
                embed["fields"] = self.format_data_discord(data)
            
            payload = {
                "embeds": [embed]
            }
            
            response = requests.post(discord_config['webhook_url'], json=payload)
            response.raise_for_status()
            
            logger.info("Discord alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
    
    def format_data_html(self, data: Dict) -> str:
        """Format data for HTML email"""
        html = "<table style='border-collapse: collapse; width: 100%;'>"
        for key, value in data.items():
            html += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid #ddd; font-weight: bold;'>{key}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{value}</td>
            </tr>
            """
        html += "</table>"
        return html
    
    def format_data_telegram(self, data: Dict) -> str:
        """Format data for Telegram"""
        lines = []
        for key, value in data.items():
            lines.append(f"*{key}:* {value}")
        return "\n".join(lines)
    
    def format_data_discord(self, data: Dict) -> List[Dict]:
        """Format data for Discord"""
        fields = []
        for key, value in data.items():
            fields.append({
                "name": key,
                "value": str(value),
                "inline": True
            })
        return fields
    
    # Convenience methods for common alerts
    
    def alert_trade_opened(self, trade: Dict):
        """Alert when a trade is opened"""
        title = f"Trade Opened: {trade['symbol']}"
        message = f"{trade['side'].upper()} position opened on {trade['symbol']}"
        
        data = {
            "Symbol": trade['symbol'],
            "Side": trade['side'].upper(),
            "Entry Price": f"${trade['entry_price']:.4f}",
            "Quantity": f"{trade['quantity']:.4f}",
            "Leverage": f"{trade['leverage']}x",
            "Stop Loss": f"${trade['stop_loss']:.4f}",
            "Take Profit": f"${trade['take_profit']:.4f}",
            "Pattern": trade.get('pattern', 'N/A'),
            "Confidence": f"{trade.get('confidence', 0):.1f}%"
        }
        
        self.send_alert('trade_opened', title, message, data)
    
    def alert_trade_closed(self, trade: Dict):
        """Alert when a trade is closed"""
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_percentage', 0)
        
        title = f"Trade Closed: {trade['symbol']} ({'+' if pnl >= 0 else ''}{pnl_pct:.2f}%)"
        message = f"{trade['side'].upper()} position closed on {trade['symbol']}"
        
        data = {
            "Symbol": trade['symbol'],
            "Side": trade['side'].upper(),
            "Entry Price": f"${trade['entry_price']:.4f}",
            "Exit Price": f"${trade.get('exit_price', 0):.4f}",
            "P&L": f"${pnl:+.2f}",
            "P&L %": f"{pnl_pct:+.2f}%",
            "Exit Reason": trade.get('exit_reason', 'manual'),
            "Pattern": trade.get('pattern', 'N/A')
        }
        
        self.send_alert('trade_closed', title, message, data)
    
    def alert_high_confidence_signal(self, signal: Dict):
        """Alert for high confidence trading signals"""
        title = f"High Confidence Signal: {signal['symbol']}"
        message = f"{signal['direction'].upper()} signal detected with {signal['confidence']:.1f}% confidence"
        
        data = {
            "Symbol": signal['symbol'],
            "Direction": signal['direction'].upper(),
            "Pattern": signal.get('pattern', 'N/A'),
            "Confidence": f"{signal['confidence']:.1f}%",
            "Entry Price": f"${signal['entry_price']:.4f}",
            "Stop Loss": f"${signal['stop_loss']:.4f}",
            "Take Profit": f"${signal['take_profit']:.4f}",
            "R/R Ratio": f"{signal.get('risk_reward_ratio', 0):.1f}:1"
        }
        
        self.send_alert('high_confidence_signal', title, message, data)
    
    def alert_daily_summary(self, stats: Dict):
        """Send daily performance summary"""
        title = "Daily Trading Summary"
        message = f"Performance summary for {datetime.now().strftime('%Y-%m-%d')}"
        
        data = {
            "Total Trades": stats.get('total_trades', 0),
            "Winning Trades": stats.get('winning_trades', 0),
            "Losing Trades": stats.get('losing_trades', 0),
            "Win Rate": f"{stats.get('win_rate', 0):.1f}%",
            "Total P&L": f"${stats.get('total_pnl', 0):+.2f}",
            "ROI": f"{stats.get('roi', 0):+.2f}%",
            "Current Balance": f"${stats.get('current_balance', 0):,.2f}",
            "Open Positions": stats.get('open_positions', 0)
        }
        
        self.send_alert('daily_summary', title, message, data)
    
    def alert_system_error(self, error_message: str, details: Optional[str] = None):
        """Alert for system errors"""
        title = "System Error"
        message = f"An error occurred in BubbyBot: {error_message}"
        
        data = {}
        if details:
            data["Details"] = details
        data["Timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.send_alert('system_error', title, message, data)

# Example usage
if __name__ == "__main__":
    alert_system = AlertSystem()
    
    # Test alert
    test_trade = {
        'symbol': 'AVAX-USD',
        'side': 'long',
        'entry_price': 14.82,
        'quantity': 100,
        'leverage': 5.0,
        'stop_loss': 14.50,
        'take_profit': 15.50,
        'pattern': 'green_dot',
        'confidence': 89.0
    }
    
    alert_system.alert_trade_opened(test_trade)


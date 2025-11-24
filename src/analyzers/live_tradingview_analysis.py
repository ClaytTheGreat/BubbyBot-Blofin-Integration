"""
Live TradingView Analysis Script
Real-time Market Cipher and Lux Algo analysis using TradingView API
"""

import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingview_integration import (
    initialize_tradingview_api,
    get_live_market_cipher_data,
    get_live_lux_algo_data,
    get_live_comprehensive_analysis
)
from enhanced_signal_processor import process_enhanced_trading_signal
from risk_management_system import validate_trading_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tradingview_config() -> Dict[str, Any]:
    """Load TradingView configuration"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'tradingview_config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading TradingView config: {e}")
        return {}

def setup_tradingview_connection(username: str, password: str) -> bool:
    """Setup TradingView API connection"""
    try:
        logger.info("Setting up TradingView API connection...")
        success = initialize_tradingview_api(username, password)
        
        if success:
            logger.info("✅ TradingView API connection established successfully")
            return True
        else:
            logger.error("❌ Failed to establish TradingView API connection")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up TradingView connection: {e}")
        return False

def get_live_analysis(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    """Get live analysis from TradingView indicators"""
    try:
        logger.info(f"🔍 Getting live analysis for {symbol} ({timeframe})")
        
        # Get comprehensive analysis from TradingView
        analysis = get_live_comprehensive_analysis(symbol, timeframe)
        
        if not analysis:
            logger.error("Failed to get TradingView analysis")
            return {}
        
        # Get individual indicator data
        market_cipher = get_live_market_cipher_data(symbol, timeframe)
        lux_algo = get_live_lux_algo_data(symbol, timeframe)
        
        # Combine all data
        combined_analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'tradingview_analysis': analysis,
            'market_cipher_raw': market_cipher.__dict__ if market_cipher else {},
            'lux_algo_raw': lux_algo.__dict__ if lux_algo else {},
            'data_source': 'TradingView Live API'
        }
        
        logger.info(f"✅ Live analysis retrieved for {symbol}")
        return combined_analysis
        
    except Exception as e:
        logger.error(f"Error getting live analysis: {e}")
        return {}

def run_live_avax_analysis(username: str, password: str) -> Dict[str, Any]:
    """Run live AVAX analysis using TradingView API"""
    try:
        print("🚀 AI TRADING BOT - LIVE TRADINGVIEW ANALYSIS")
        print("=" * 60)
        print()
        
        # Setup TradingView connection
        if not setup_tradingview_connection(username, password):
            print("❌ Failed to connect to TradingView. Please check your credentials.")
            return {}
        
        print("🔗 Connected to TradingView API")
        print("📊 Fetching live Market Cipher and Lux Algo data...")
        print()
        
        # Get live analysis for AVAX
        analysis = get_live_analysis("AVAXUSDT", "1h")
        
        if not analysis:
            print("❌ Failed to retrieve live analysis data")
            return {}
        
        # Display results
        print("✅ LIVE TRADINGVIEW ANALYSIS RESULTS:")
        print("-" * 50)
        print(f"Symbol: {analysis['symbol']}")
        print(f"Timeframe: {analysis['timeframe']}")
        print(f"Timestamp: {analysis['timestamp']}")
        print(f"Data Source: {analysis['data_source']}")
        print()
        
        # Market Cipher Analysis
        if 'market_cipher_raw' in analysis and analysis['market_cipher_raw']:
            mc_data = analysis['market_cipher_raw']
            print("📈 LIVE MARKET CIPHER ANALYSIS:")
            print(f"   Money Flow Signal: {mc_data.get('money_flow_signal', 'N/A').upper()}")
            print(f"   Momentum Signal: {mc_data.get('momentum_signal', 'N/A').upper()}")
            print(f"   Squeeze Active: {'YES' if mc_data.get('squeeze_momentum', False) else 'NO'}")
            print(f"   Divergence Detected: {'YES' if mc_data.get('divergence_detected', False) else 'NO'}")
            print(f"   Wave Trend: {mc_data.get('wave_trend', 0.0)}")
            print(f"   Support Level: ${mc_data.get('support_level', 0.0):.2f}")
            print(f"   Resistance Level: ${mc_data.get('resistance_level', 0.0):.2f}")
            print()
        
        # Lux Algo Analysis
        if 'lux_algo_raw' in analysis and analysis['lux_algo_raw']:
            lux_data = analysis['lux_algo_raw']
            print("🏗️ LIVE LUX ALGO ANALYSIS:")
            print(f"   Market Structure: {lux_data.get('market_structure', 'N/A').upper()}")
            print(f"   Smart Money Signal: {lux_data.get('smart_money_signal', 'N/A').upper()}")
            print(f"   Bullish Order Block: {'ACTIVE' if lux_data.get('bullish_order_block', False) else 'INACTIVE'}")
            print(f"   Bearish Order Block: {'ACTIVE' if lux_data.get('bearish_order_block', False) else 'INACTIVE'}")
            print(f"   Premium Zone: {'YES' if lux_data.get('premium_zone', False) else 'NO'}")
            print(f"   Discount Zone: {'YES' if lux_data.get('discount_zone', False) else 'NO'}")
            print(f"   Structure Break: {'YES' if lux_data.get('structure_break', False) else 'NO'}")
            print(f"   Trend Strength: {lux_data.get('trend_strength', 0.0):.2f}")
            print()
        
        # TradingView Analysis Summary
        if 'tradingview_analysis' in analysis and analysis['tradingview_analysis']:
            tv_analysis = analysis['tradingview_analysis']
            print("🎯 TRADINGVIEW CONFLUENCE ANALYSIS:")
            print(f"   Confluence Score: {tv_analysis.get('confluence_score', 0.0):.3f}")
            print(f"   Overall Signal: {tv_analysis.get('overall_signal', 'neutral').upper()}")
            
            if 'market_cipher' in tv_analysis:
                mc = tv_analysis['market_cipher']
                print(f"   MC Money Flow: {mc.get('money_flow_signal', 'N/A').upper()}")
                print(f"   MC Momentum: {mc.get('momentum_signal', 'N/A').upper()}")
            
            if 'lux_algo' in tv_analysis:
                lux = tv_analysis['lux_algo']
                print(f"   Lux Structure: {lux.get('market_structure', 'N/A').upper()}")
                print(f"   Lux Smart Money: {lux.get('smart_money_signal', 'N/A').upper()}")
            print()
        
        # AI Bot Processing
        print("🤖 PROCESSING THROUGH AI TRADING BOT...")
        try:
            # Convert to format expected by our AI systems
            signal_data = {
                'symbol': analysis['symbol'],
                'timeframe': analysis['timeframe'],
                'signal_type': 'live_analysis',
                'confidence': tv_analysis.get('confluence_score', 0.5) if 'tradingview_analysis' in analysis else 0.5,
                'market_cipher': analysis.get('market_cipher_raw', {}),
                'lux_algo': analysis.get('lux_algo_raw', {}),
                'timestamp': analysis['timestamp']
            }
            
            # Process through enhanced signal processor
            enhanced_result = process_enhanced_trading_signal(signal_data)
            
            if enhanced_result:
                signal = enhanced_result['signal']
                print("✅ AI PROCESSING COMPLETE")
                print()
                print("🎯 AI ENHANCED ANALYSIS:")
                print(f"   AI Confidence: {signal['confidence']:.3f}")
                print(f"   Signal Strength: {signal['signal_strength']:.3f}")
                print(f"   Execution Priority: {signal['execution_priority']}/10")
                print(f"   Risk/Reward: {signal['risk_reward_ratio']:.2f}")
                print()
                
                # Risk validation
                validation = validate_trading_signal(signal)
                print("🛡️ RISK VALIDATION:")
                print(f"   Validation Status: {validation.get('status', 'unknown').upper()}")
                print(f"   Risk Score: {validation.get('risk_score', 0.0):.3f}")
                if 'recommendations' in validation:
                    print(f"   Recommendations: {', '.join(validation['recommendations'])}")
                print()
            
        except Exception as e:
            print(f"⚠️ AI processing error: {e}")
        
        print("=" * 60)
        print("📝 Live TradingView analysis completed")
        print("⚠️ This uses real TradingView data - Not financial advice")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in live AVAX analysis: {e}")
        print(f"❌ Analysis error: {e}")
        return {}

def main():
    """Main function for live TradingView analysis"""
    try:
        # Load configuration
        config = load_tradingview_config()
        
        if not config:
            print("❌ Failed to load TradingView configuration")
            return
        
        # Get credentials (you'll need to update these)
        username = config.get('tradingview', {}).get('username', 'YOUR_USERNAME')
        password = config.get('tradingview', {}).get('password', 'YOUR_PASSWORD')
        
        if username == 'YOUR_USERNAME' or password == 'YOUR_PASSWORD':
            print("⚠️ Please update your TradingView credentials in tradingview_config.json")
            print("   Username and password are required for live data access")
            return
        
        # Run live analysis
        analysis = run_live_avax_analysis(username, password)
        
        if analysis:
            # Save analysis to file
            output_file = f"live_avax_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"📄 Analysis saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Main error: {e}")

if __name__ == "__main__":
    main()

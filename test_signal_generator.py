"""
Test script for signal generator
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 60)
    print("Testing BubbyBot Signal Generator")
    print("=" * 60)
    
    # Create signal generator
    config = {
        'min_confluence': 0.70,
        'mc_weight': 0.5,
        'lux_weight': 0.5
    }
    
    generator = SignalGenerator(config)
    
    # Test with BTC
    print("\n🔍 Generating signal for BTC-USDT...")
    signal = generator.generate_signal('BTC-USDT')
    
    if signal:
        print("\n✅ Trading Signal Generated:")
        print(f"  Instrument: {signal.instrument}")
        print(f"  Side: {signal.side.upper()}")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Entry: ${signal.entry_price:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Pattern: {signal.pattern_type}")
        if signal.metadata:
            print(f"\n  Metadata:")
            for key, value in signal.metadata.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
    else:
        print("\n❌ No signal generated")
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


if __name__ == "__main__":
    main()

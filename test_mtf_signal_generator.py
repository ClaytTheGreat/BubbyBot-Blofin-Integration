"""
Test Multi-Timeframe Signal Generator
Tests MTF analysis across all timeframes
"""

import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.signal_generator_mtf import MTFSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_mtf_signal_generation():
    """Test MTF signal generation"""
    
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME SIGNAL GENERATOR TEST")
    print("=" * 80)
    
    # Configuration
    config = {
        'scalping_mode': True,
        'min_confidence': 0.70,
        'min_alignment': 0.65,
        'require_trend_alignment': True
    }
    
    print(f"\nConfiguration:")
    print(f"  Scalping Mode: {config['scalping_mode']}")
    print(f"  Min Confidence: {config['min_confidence']:.0%}")
    print(f"  Min Alignment: {config['min_alignment']:.0%}")
    print(f"  Require Trend Alignment: {config['require_trend_alignment']}")
    
    # Initialize generator
    print(f"\nInitializing MTF Signal Generator...")
    generator = MTFSignalGenerator(config)
    
    # Test instruments
    instruments = ['BTC-USDT', 'ETH-USDT']
    
    print(f"\n" + "=" * 80)
    print(f"Testing {len(instruments)} instruments...")
    print(f"=" * 80)
    
    results = []
    
    for instrument in instruments:
        print(f"\n{'=' * 80}")
        print(f"Testing: {instrument}")
        print(f"{'=' * 80}")
        
        try:
            # Generate signal
            signal = generator.generate_signal(instrument)
            
            if signal:
                print(f"\n✅ SIGNAL GENERATED")
                print(f"{'=' * 80}")
                print(f"Instrument: {signal.instrument}")
                print(f"Side: {signal.side.upper()}")
                print(f"Confidence: {signal.confidence:.2%}")
                print(f"Entry Price: ${signal.entry_price:.2f}")
                print(f"Stop Loss: ${signal.stop_loss:.2f}")
                print(f"Take Profit: ${signal.take_profit:.2f}")
                print(f"Pattern: {signal.pattern_type}")
                print(f"Timestamp: {signal.timestamp}")
                
                print(f"\nMetadata:")
                print(f"  Entry Timeframe: {signal.metadata.get('entry_timeframe', 'N/A')}")
                print(f"  Trend Alignment: {signal.metadata.get('trend_alignment', 0):.2%}")
                print(f"  Momentum Alignment: {signal.metadata.get('momentum_alignment', 0):.2%}")
                print(f"  Overall Alignment: {signal.metadata.get('overall_alignment', 0):.2%}")
                print(f"  Micro Signals: {signal.metadata.get('micro_signals_count', 0)}")
                print(f"  Scalp Signals: {signal.metadata.get('scalp_signals_count', 0)}")
                print(f"  Swing Signals: {signal.metadata.get('swing_signals_count', 0)}")
                print(f"  MTF Enabled: {signal.metadata.get('mtf_enabled', False)}")
                print(f"  Scalping Mode: {signal.metadata.get('scalping_mode', False)}")
                
                print(f"\nTimeframe Breakdown:")
                for key, value in signal.metadata.items():
                    if '_direction' in key:
                        tf = key.replace('_direction', '')
                        confidence = signal.metadata.get(f'{tf}_confidence', 0)
                        print(f"  {tf}: {value.upper()} ({confidence:.2%})")
                
                print(f"\nAnalysis Summary:")
                print(f"  {signal.metadata.get('analysis_summary', 'N/A')}")
                
                # Calculate risk/reward
                if signal.side == 'buy':
                    risk = signal.entry_price - signal.stop_loss
                    reward = signal.take_profit - signal.entry_price
                else:
                    risk = signal.stop_loss - signal.entry_price
                    reward = signal.entry_price - signal.take_profit
                
                rr_ratio = reward / risk if risk > 0 else 0
                
                print(f"\nRisk/Reward:")
                print(f"  Risk: ${risk:.2f} ({(risk/signal.entry_price)*100:.2f}%)")
                print(f"  Reward: ${reward:.2f} ({(reward/signal.entry_price)*100:.2f}%)")
                print(f"  R:R Ratio: {rr_ratio:.2f}:1")
                
                results.append({
                    'instrument': instrument,
                    'signal': signal,
                    'success': True
                })
                
            else:
                print(f"\n❌ NO SIGNAL GENERATED")
                print(f"  No confluence or alignment threshold not met")
                
                results.append({
                    'instrument': instrument,
                    'signal': None,
                    'success': False
                })
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            logger.exception(f"Error testing {instrument}")
            
            results.append({
                'instrument': instrument,
                'signal': None,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"TEST SUMMARY")
    print(f"=" * 80)
    
    total = len(results)
    signals_generated = sum(1 for r in results if r['signal'] is not None)
    errors = sum(1 for r in results if 'error' in r)
    
    print(f"\nResults:")
    print(f"  Total Instruments: {total}")
    print(f"  Signals Generated: {signals_generated}")
    print(f"  No Signal: {total - signals_generated - errors}")
    print(f"  Errors: {errors}")
    
    if signals_generated > 0:
        print(f"\nSignal Rate: {(signals_generated/total)*100:.1f}%")
        
        # Show all signals
        print(f"\nGenerated Signals:")
        for r in results:
            if r['signal']:
                sig = r['signal']
                print(f"  {sig.instrument}: {sig.side.upper()} @ ${sig.entry_price:.2f} "
                      f"({sig.confidence:.0%} confidence, {sig.pattern_type})")
    
    print(f"\n" + "=" * 80)
    print(f"TEST COMPLETE")
    print(f"=" * 80)
    
    return results


if __name__ == "__main__":
    try:
        results = test_mtf_signal_generation()
        
        # Exit code based on results
        if any(r['signal'] is not None for r in results):
            print(f"\n✅ Test passed: At least one signal generated")
            sys.exit(0)
        else:
            print(f"\n⚠️  Test completed: No signals generated (may be normal)")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test failed")
        sys.exit(1)

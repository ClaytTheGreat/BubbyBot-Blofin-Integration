"""
Unified Signal Generator
Combines Market Cipher and Lux Algo analysis for confluence-based signal generation
"""

import logging
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from core.market_cipher_analyzer import MarketCipherAnalyzer, MarketCipherSignal
from core.lux_algo_analyzer import LuxAlgoAnalyzer, LuxAlgoSignal
from blofin.exchange_adapter import TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceScore:
    """Confluence scoring result"""
    overall_score: float  # 0-1
    mc_signal: Optional[MarketCipherSignal]
    lux_signal: Optional[LuxAlgoSignal]
    direction: str  # 'buy' or 'sell'
    confidence: float
    components: Dict[str, float]


class SignalGenerator:
    """
    Unified Signal Generator
    Combines Market Cipher and Lux Algo for high-confidence signals
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Signal Generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize analyzers
        self.mc_analyzer = MarketCipherAnalyzer(config)
        self.lux_analyzer = LuxAlgoAnalyzer(config)
        
        # Confluence settings
        self.min_confluence = self.config.get('min_confluence', 0.70)
        self.mc_weight = self.config.get('mc_weight', 0.5)
        self.lux_weight = self.config.get('lux_weight', 0.5)
        
        # Timeframes to analyze
        self.timeframes = self.config.get('timeframes', ['15m'])
        
        logger.info("Signal Generator initialized")
    
    def generate_signal(self, instrument: str) -> Optional[TradingSignal]:
        """
        Generate trading signal for instrument
        
        Args:
            instrument: Trading pair (e.g., 'BTC-USDT')
            
        Returns:
            TradingSignal or None
        """
        try:
            logger.info(f"Generating signal for {instrument}")
            
            # Get market data
            data = self._fetch_market_data(instrument)
            if data is None or len(data) < 200:
                logger.warning(f"Insufficient data for {instrument}")
                return None
            
            # Analyze with both systems
            mc_signal = self.mc_analyzer.analyze(data, timeframe="15m")
            lux_signal = self.lux_analyzer.analyze(data, timeframe="15m")
            
            # Calculate confluence
            confluence = self._calculate_confluence(mc_signal, lux_signal)
            
            if confluence is None:
                logger.info(f"No confluence signal for {instrument}")
                return None
            
            if confluence.overall_score < self.min_confluence:
                logger.info(f"Confluence score too low: {confluence.overall_score:.2%}")
                return None
            
            # Convert to TradingSignal
            trading_signal = self._create_trading_signal(
                instrument=instrument,
                confluence=confluence,
                current_price=data['close'].iloc[-1]
            )
            
            logger.info(f"✅ Generated {trading_signal.side} signal for {instrument} "
                       f"with {trading_signal.confidence:.2%} confidence")
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {instrument}: {e}")
            return None
    
    def _fetch_market_data(self, instrument: str) -> Optional[pd.DataFrame]:
        """Fetch market data from Yahoo Finance"""
        try:
            # Convert Blofin format to Yahoo format
            symbol = instrument.replace('-', '')
            if symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '-USD')
            
            logger.debug(f"Fetching data for {symbol}")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo", interval="15m")
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            logger.debug(f"Fetched {len(data)} bars for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {instrument}: {e}")
            return None
    
    def _calculate_confluence(
        self,
        mc_signal: Optional[MarketCipherSignal],
        lux_signal: Optional[LuxAlgoSignal]
    ) -> Optional[ConfluenceScore]:
        """Calculate confluence between Market Cipher and Lux Algo signals"""
        try:
            # Need at least one signal
            if mc_signal is None and lux_signal is None:
                return None
            
            # Determine direction
            mc_direction = mc_signal.direction if mc_signal else None
            lux_direction = lux_signal.direction if lux_signal else None
            
            # Check for conflicting signals
            if mc_direction and lux_direction and mc_direction != lux_direction:
                logger.info("Conflicting signals: MC={}, Lux={}".format(
                    mc_direction, lux_direction))
                return None
            
            # Determine final direction
            direction = mc_direction or lux_direction
            
            # Calculate component scores
            components = {}
            
            if mc_signal:
                components['market_cipher'] = mc_signal.confidence
            else:
                components['market_cipher'] = 0.5  # Neutral
            
            if lux_signal:
                components['lux_algo'] = lux_signal.confidence
            else:
                components['lux_algo'] = 0.5  # Neutral
            
            # Calculate weighted overall score
            overall_score = (
                components['market_cipher'] * self.mc_weight +
                components['lux_algo'] * self.lux_weight
            )
            
            # Bonus for both signals agreeing
            if mc_signal and lux_signal and mc_direction == lux_direction:
                overall_score = min(1.0, overall_score * 1.15)  # 15% bonus
                components['confluence_bonus'] = 0.15
            
            # Calculate final confidence
            confidence = overall_score
            
            confluence = ConfluenceScore(
                overall_score=overall_score,
                mc_signal=mc_signal,
                lux_signal=lux_signal,
                direction=direction,
                confidence=confidence,
                components=components
            )
            
            logger.info(f"Confluence: {direction} signal with {overall_score:.2%} score")
            logger.info(f"  Market Cipher: {components['market_cipher']:.2%}")
            logger.info(f"  Lux Algo: {components['lux_algo']:.2%}")
            
            return confluence
            
        except Exception as e:
            logger.error(f"Error calculating confluence: {e}")
            return None
    
    def _create_trading_signal(
        self,
        instrument: str,
        confluence: ConfluenceScore,
        current_price: float
    ) -> TradingSignal:
        """Create TradingSignal from confluence analysis"""
        
        # Calculate stop loss and take profit
        if confluence.direction == 'buy':
            # Use Lux Algo levels if available, otherwise percentage-based
            if confluence.lux_signal and confluence.lux_signal.stop_loss:
                stop_loss = confluence.lux_signal.stop_loss
            else:
                stop_loss = current_price * 0.98  # 2% stop loss
            
            if confluence.lux_signal and confluence.lux_signal.take_profit:
                take_profit = confluence.lux_signal.take_profit
            else:
                take_profit = current_price * 1.06  # 6% take profit
        
        else:  # sell
            if confluence.lux_signal and confluence.lux_signal.stop_loss:
                stop_loss = confluence.lux_signal.stop_loss
            else:
                stop_loss = current_price * 1.02  # 2% stop loss
            
            if confluence.lux_signal and confluence.lux_signal.take_profit:
                take_profit = confluence.lux_signal.take_profit
            else:
                take_profit = current_price * 0.94  # 6% take profit
        
        # Build description
        description_parts = []
        if confluence.mc_signal:
            description_parts.append(f"MC: {confluence.mc_signal.description}")
        if confluence.lux_signal:
            description_parts.append(f"Lux: {confluence.lux_signal.description}")
        
        description = " | ".join(description_parts)
        
        # Determine pattern type
        if confluence.mc_signal and confluence.lux_signal:
            pattern_type = "confluence"
        elif confluence.mc_signal:
            pattern_type = confluence.mc_signal.signal_type.value
        else:
            pattern_type = confluence.lux_signal.signal_type.value
        
        return TradingSignal(
            instrument=instrument,
            side=confluence.direction,
            confidence=confluence.confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe="15m",
            pattern_type=pattern_type,
            timestamp=datetime.now(),
            metadata={
                'mc_confidence': confluence.components.get('market_cipher', 0),
                'lux_confidence': confluence.components.get('lux_algo', 0),
                'confluence_score': confluence.overall_score,
                'description': description
            }
        )
    
    def analyze_multiple_timeframes(self, instrument: str) -> Optional[TradingSignal]:
        """
        Analyze across multiple timeframes for higher confidence
        
        Args:
            instrument: Trading pair
            
        Returns:
            TradingSignal or None
        """
        # TODO: Implement multi-timeframe analysis
        # For now, use single timeframe
        return self.generate_signal(instrument)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = {
        'min_confluence': 0.70,
        'mc_weight': 0.5,
        'lux_weight': 0.5
    }
    
    generator = SignalGenerator(config)
    
    # Test with BTC
    signal = generator.generate_signal('BTC-USDT')
    
    if signal:
        print(f"\n✅ Trading Signal Generated:")
        print(f"  Instrument: {signal.instrument}")
        print(f"  Side: {signal.side.upper()}")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Entry: ${signal.entry_price:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Pattern: {signal.pattern_type}")
        if signal.metadata:
            print(f"  Description: {signal.metadata.get('description', 'N/A')}")
    else:
        print("\n❌ No signal generated")

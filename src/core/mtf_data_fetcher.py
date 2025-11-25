"""
Multi-Timeframe Data Fetcher
Fetches OHLCV data across multiple timeframes for MTF analysis
"""

import logging
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MTFDataFetcher:
    """
    Multi-Timeframe Data Fetcher
    Fetches and manages data across multiple timeframes
    """
    
    # Timeframe mapping: internal -> yfinance
    TIMEFRAME_MAPPING = {
        # Micro timeframes (simulated from 1m data)
        '1s': '1m',
        '5s': '1m',
        '10s': '1m',
        '15s': '1m',
        '30s': '1m',
        
        # Standard timeframes
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '1d',  # Approximate with daily data
    }
    
    # Data period requirements
    PERIOD_REQUIREMENTS = {
        '1m': '1d',   # 1 day of 1-minute data
        '5m': '5d',   # 5 days of 5-minute data
        '15m': '1mo', # 1 month of 15-minute data
        '1h': '3mo',  # 3 months of hourly data
        '4h': '1y',   # 1 year of daily data (for 4h simulation)
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize MTF Data Fetcher
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.cache = {}  # Cache for fetched data
        self.cache_expiry = {}  # Cache expiry times
        self.cache_duration = self.config.get('cache_duration', 60)  # 60 seconds default
        
        logger.info("MTF Data Fetcher initialized")
    
    def fetch_all_timeframes(
        self, 
        instrument: str, 
        timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all specified timeframes
        
        Args:
            instrument: Trading pair (e.g., 'BTC-USDT')
            timeframes: List of timeframes to fetch
            
        Returns:
            Dictionary of {timeframe: DataFrame}
        """
        try:
            logger.info(f"Fetching data for {instrument} across {len(timeframes)} timeframes")
            
            data_dict = {}
            
            # Convert instrument format
            symbol = self._convert_instrument(instrument)
            
            # Fetch base timeframes
            base_data = self._fetch_base_timeframes(symbol)
            
            # Process each requested timeframe
            for tf in timeframes:
                try:
                    data = self._get_timeframe_data(tf, base_data)
                    if data is not None and len(data) > 0:
                        data_dict[tf] = data
                        logger.debug(f"  {tf}: {len(data)} bars")
                    else:
                        logger.warning(f"  {tf}: No data available")
                except Exception as e:
                    logger.error(f"  {tf}: Error - {e}")
            
            logger.info(f"Successfully fetched {len(data_dict)}/{len(timeframes)} timeframes")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error fetching MTF data for {instrument}: {e}")
            return {}
    
    def _convert_instrument(self, instrument: str) -> str:
        """Convert Blofin format to Yahoo Finance format"""
        symbol = instrument.replace('-', '')
        if symbol.endswith('USDT'):
            symbol = symbol.replace('USDT', '-USD')
        return symbol
    
    def _fetch_base_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch base timeframes from data source"""
        base_data = {}
        
        # Check cache first
        cache_key = f"{symbol}_base"
        if self._is_cached(cache_key):
            logger.debug("Using cached base data")
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch each base timeframe
            for yf_interval, period in self.PERIOD_REQUIREMENTS.items():
                try:
                    logger.debug(f"Fetching {yf_interval} data (period={period})")
                    data = ticker.history(period=period, interval=yf_interval)
                    
                    if not data.empty:
                        # Standardize column names
                        data.columns = data.columns.str.lower()
                        base_data[yf_interval] = data
                        logger.debug(f"  Fetched {len(data)} bars for {yf_interval}")
                    else:
                        logger.warning(f"  No data returned for {yf_interval}")
                        
                except Exception as e:
                    logger.error(f"  Error fetching {yf_interval}: {e}")
            
            # Cache the results
            self.cache[cache_key] = base_data
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
            
        except Exception as e:
            logger.error(f"Error fetching base data for {symbol}: {e}")
        
        return base_data
    
    def _get_timeframe_data(self, timeframe: str, base_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Get data for specific timeframe"""
        
        # Map to base timeframe
        base_tf = self.TIMEFRAME_MAPPING.get(timeframe)
        if not base_tf:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return None
        
        # Get base data
        if base_tf not in base_data:
            logger.warning(f"Base timeframe {base_tf} not available")
            return None
        
        data = base_data[base_tf].copy()
        
        # For micro timeframes, resample from 1m data
        if timeframe in ['1s', '5s', '10s', '15s', '30s']:
            data = self._simulate_micro_timeframe(data, timeframe)
        
        # For 4h, resample from daily data
        elif timeframe == '4h':
            data = self._resample_to_4h(data)
        
        return data
    
    def _simulate_micro_timeframe(self, data_1m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Simulate micro timeframe data from 1-minute data
        Note: This is an approximation for backtesting purposes
        """
        try:
            # Extract seconds from timeframe
            seconds = int(target_tf.replace('s', ''))
            
            # For simulation, we'll create synthetic bars by splitting 1m candles
            # This is not perfect but gives us data to work with
            
            simulated_data = []
            
            for idx, row in data_1m.iterrows():
                # Split 1m candle into smaller pieces
                num_pieces = 60 // seconds
                
                for i in range(num_pieces):
                    # Create synthetic OHLC
                    # This is a simplified simulation
                    piece = {
                        'open': row['open'] if i == 0 else simulated_data[-1]['close'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'] if i == num_pieces - 1 else row['open'] + (row['close'] - row['open']) * (i + 1) / num_pieces,
                        'volume': row['volume'] / num_pieces
                    }
                    simulated_data.append(piece)
            
            df = pd.DataFrame(simulated_data)
            
            # Limit to last 500 bars to avoid too much data
            if len(df) > 500:
                df = df.tail(500)
            
            logger.debug(f"Simulated {len(df)} bars for {target_tf}")
            return df
            
        except Exception as e:
            logger.error(f"Error simulating {target_tf}: {e}")
            return data_1m  # Fallback to 1m data
    
    def _resample_to_4h(self, data_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily data to approximate 4-hour timeframe
        Note: This is an approximation
        """
        try:
            # For simplicity, we'll use daily data as proxy for 4h
            # In production, you'd want actual 4h data
            return data_daily
            
        except Exception as e:
            logger.error(f"Error resampling to 4h: {e}")
            return data_daily
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and not expired"""
        if cache_key not in self.cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        if datetime.now() > self.cache_expiry[cache_key]:
            # Expired
            del self.cache[cache_key]
            del self.cache_expiry[cache_key]
            return False
        
        return True
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("Cache cleared")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fetcher = MTFDataFetcher()
    
    timeframes = ['1s', '5s', '15s', '30s', '1m', '5m', '15m', '1h', '4h']
    data_dict = fetcher.fetch_all_timeframes('BTC-USDT', timeframes)
    
    print(f"\n✅ Fetched data for {len(data_dict)} timeframes:")
    for tf, data in data_dict.items():
        print(f"  {tf}: {len(data)} bars")

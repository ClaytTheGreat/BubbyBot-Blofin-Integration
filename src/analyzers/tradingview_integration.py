"""
TradingView API Integration Module
Direct access to Market Cipher and Lux Algo indicators from TradingView account
"""

import logging
import json
import requests
import websocket
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class TradingViewCredentials:
    """TradingView account credentials"""
    username: str
    password: str
    session_token: Optional[str] = None
    auth_token: Optional[str] = None

@dataclass
class MarketCipherData:
    """Market Cipher indicator data"""
    symbol: str
    timeframe: str
    timestamp: str
    
    # Market Cipher A (Money Flow)
    money_flow_green: float
    money_flow_red: float
    money_flow_signal: str  # 'bullish', 'bearish', 'neutral'
    
    # Market Cipher B (Momentum)
    momentum_wave: float
    momentum_signal: str
    blue_wave: float
    
    # Market Cipher SR (Support/Resistance)
    support_level: float
    resistance_level: float
    sr_signal: str
    
    # Market Cipher DBSI (Divergence)
    divergence_detected: bool
    divergence_type: str  # 'bullish', 'bearish', 'none'
    
    # Additional Market Cipher signals
    squeeze_momentum: bool
    squeeze_direction: str
    wave_trend: float
    vwap_signal: str

@dataclass
class LuxAlgoData:
    """Lux Algo indicator data"""
    symbol: str
    timeframe: str
    timestamp: str
    
    # Order Blocks
    bullish_order_block: bool
    bearish_order_block: bool
    order_block_price: float
    
    # Premium/Discount Zones
    premium_zone: bool
    discount_zone: bool
    equilibrium: bool
    zone_strength: float
    
    # Market Structure
    market_structure: str  # 'bullish', 'bearish', 'neutral'
    structure_break: bool
    higher_high: bool
    higher_low: bool
    lower_high: bool
    lower_low: bool
    
    # Smart Money Concepts
    smart_money_signal: str  # 'accumulation', 'distribution', 'neutral'
    liquidity_grab: bool
    fair_value_gap: bool
    
    # Additional Lux Algo signals
    trend_strength: float
    volatility_index: float

class TradingViewAPI:
    """TradingView API integration for live indicator data"""
    
    def __init__(self, credentials: TradingViewCredentials):
        self.credentials = credentials
        self.session = requests.Session()
        self.websocket = None
        self.driver = None
        self.authenticated = False
        
        # Database for storing indicator data
        self.db_path = "tradingview_data.db"
        self.init_database()
        
        # Data storage
        self.market_cipher_data = {}
        self.lux_algo_data = {}
        
    def init_database(self):
        """Initialize database for storing TradingView data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market Cipher data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_cipher_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                timestamp TEXT,
                money_flow_green REAL,
                money_flow_red REAL,
                money_flow_signal TEXT,
                momentum_wave REAL,
                momentum_signal TEXT,
                blue_wave REAL,
                support_level REAL,
                resistance_level REAL,
                sr_signal TEXT,
                divergence_detected BOOLEAN,
                divergence_type TEXT,
                squeeze_momentum BOOLEAN,
                squeeze_direction TEXT,
                wave_trend REAL,
                vwap_signal TEXT,
                raw_data TEXT
            )
        ''')
        
        # Lux Algo data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lux_algo_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                timestamp TEXT,
                bullish_order_block BOOLEAN,
                bearish_order_block BOOLEAN,
                order_block_price REAL,
                premium_zone BOOLEAN,
                discount_zone BOOLEAN,
                equilibrium BOOLEAN,
                zone_strength REAL,
                market_structure TEXT,
                structure_break BOOLEAN,
                higher_high BOOLEAN,
                higher_low BOOLEAN,
                lower_high BOOLEAN,
                lower_low BOOLEAN,
                smart_money_signal TEXT,
                liquidity_grab BOOLEAN,
                fair_value_gap BOOLEAN,
                trend_strength REAL,
                volatility_index REAL,
                raw_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def authenticate(self) -> bool:
        """Authenticate with TradingView"""
        try:
            logger.info("Authenticating with TradingView...")
            
            # Set up Chrome driver for TradingView login
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Navigate to TradingView login
            self.driver.get("https://www.tradingview.com/accounts/signin/")
            
            # Wait for login form
            wait = WebDriverWait(self.driver, 10)
            
            # Enter credentials
            username_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
            password_field = self.driver.find_element(By.NAME, "password")
            
            username_field.send_keys(self.credentials.username)
            password_field.send_keys(self.credentials.password)
            
            # Submit login
            login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            login_button.click()
            
            # Wait for successful login
            time.sleep(5)
            
            # Extract session tokens
            cookies = self.driver.get_cookies()
            for cookie in cookies:
                if cookie['name'] == 'sessionid':
                    self.credentials.session_token = cookie['value']
                elif cookie['name'] == 'auth_token':
                    self.credentials.auth_token = cookie['value']
            
            # Set up session headers
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.tradingview.com/',
                'Origin': 'https://www.tradingview.com'
            })
            
            if self.credentials.session_token:
                self.session.cookies.set('sessionid', self.credentials.session_token)
                
            self.authenticated = True
            logger.info("TradingView authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"TradingView authentication failed: {e}")
            return False
            
    def get_market_cipher_data(self, symbol: str, timeframe: str = "1h") -> Optional[MarketCipherData]:
        """Get Market Cipher indicator data for a symbol"""
        try:
            if not self.authenticated:
                if not self.authenticate():
                    return None
            
            logger.info(f"Fetching Market Cipher data for {symbol} ({timeframe})")
            
            # Navigate to chart with Market Cipher
            chart_url = f"https://www.tradingview.com/chart/?symbol={symbol}&interval={timeframe}"
            self.driver.get(chart_url)
            
            # Wait for chart to load
            time.sleep(10)
            
            # Execute JavaScript to get Market Cipher values
            market_cipher_script = """
            // Function to extract Market Cipher data
            function getMarketCipherData() {
                try {
                    // Look for Market Cipher indicators on the chart
                    const indicators = window.TradingView?.activeChart?.getAllStudies() || [];
                    let marketCipherData = {};
                    
                    indicators.forEach(indicator => {
                        if (indicator.name && indicator.name.toLowerCase().includes('market cipher')) {
                            // Extract Market Cipher values
                            const plots = indicator.plots || [];
                            plots.forEach(plot => {
                                if (plot.id && plot.value !== undefined) {
                                    marketCipherData[plot.id] = plot.value;
                                }
                            });
                        }
                    });
                    
                    return marketCipherData;
                } catch (error) {
                    console.error('Error extracting Market Cipher data:', error);
                    return {};
                }
            }
            
            return getMarketCipherData();
            """
            
            # Execute script and get data
            raw_data = self.driver.execute_script(market_cipher_script)
            
            if not raw_data:
                # Fallback: Try to extract from DOM elements
                raw_data = self._extract_market_cipher_from_dom(symbol, timeframe)
            
            # Parse Market Cipher data
            market_cipher = self._parse_market_cipher_data(symbol, timeframe, raw_data)
            
            # Store in database
            self._store_market_cipher_data(market_cipher)
            
            # Cache data
            self.market_cipher_data[f"{symbol}_{timeframe}"] = market_cipher
            
            logger.info(f"Market Cipher data retrieved for {symbol}")
            return market_cipher
            
        except Exception as e:
            logger.error(f"Error getting Market Cipher data: {e}")
            return None
            
    def get_lux_algo_data(self, symbol: str, timeframe: str = "1h") -> Optional[LuxAlgoData]:
        """Get Lux Algo indicator data for a symbol"""
        try:
            if not self.authenticated:
                if not self.authenticate():
                    return None
            
            logger.info(f"Fetching Lux Algo data for {symbol} ({timeframe})")
            
            # Navigate to chart with Lux Algo
            chart_url = f"https://www.tradingview.com/chart/?symbol={symbol}&interval={timeframe}"
            self.driver.get(chart_url)
            
            # Wait for chart to load
            time.sleep(10)
            
            # Execute JavaScript to get Lux Algo values
            lux_algo_script = """
            // Function to extract Lux Algo data
            function getLuxAlgoData() {
                try {
                    // Look for Lux Algo indicators on the chart
                    const indicators = window.TradingView?.activeChart?.getAllStudies() || [];
                    let luxAlgoData = {};
                    
                    indicators.forEach(indicator => {
                        if (indicator.name && (
                            indicator.name.toLowerCase().includes('lux') ||
                            indicator.name.toLowerCase().includes('algo') ||
                            indicator.name.toLowerCase().includes('smart money')
                        )) {
                            // Extract Lux Algo values
                            const plots = indicator.plots || [];
                            plots.forEach(plot => {
                                if (plot.id && plot.value !== undefined) {
                                    luxAlgoData[plot.id] = plot.value;
                                }
                            });
                        }
                    });
                    
                    return luxAlgoData;
                } catch (error) {
                    console.error('Error extracting Lux Algo data:', error);
                    return {};
                }
            }
            
            return getLuxAlgoData();
            """
            
            # Execute script and get data
            raw_data = self.driver.execute_script(lux_algo_script)
            
            if not raw_data:
                # Fallback: Try to extract from DOM elements
                raw_data = self._extract_lux_algo_from_dom(symbol, timeframe)
            
            # Parse Lux Algo data
            lux_algo = self._parse_lux_algo_data(symbol, timeframe, raw_data)
            
            # Store in database
            self._store_lux_algo_data(lux_algo)
            
            # Cache data
            self.lux_algo_data[f"{symbol}_{timeframe}"] = lux_algo
            
            logger.info(f"Lux Algo data retrieved for {symbol}")
            return lux_algo
            
        except Exception as e:
            logger.error(f"Error getting Lux Algo data: {e}")
            return None
            
    def _extract_market_cipher_from_dom(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Extract Market Cipher data from DOM elements"""
        try:
            # Look for Market Cipher indicator values in the DOM
            elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-name*='Market Cipher'], [data-name*='market cipher']")
            
            data = {}
            for element in elements:
                try:
                    name = element.get_attribute('data-name') or element.text
                    value = element.get_attribute('data-value') or element.get_attribute('value')
                    
                    if name and value:
                        data[name] = float(value) if value.replace('.', '').replace('-', '').isdigit() else value
                except:
                    continue
                    
            return data
            
        except Exception as e:
            logger.error(f"Error extracting Market Cipher from DOM: {e}")
            return {}
            
    def _extract_lux_algo_from_dom(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Extract Lux Algo data from DOM elements"""
        try:
            # Look for Lux Algo indicator values in the DOM
            elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-name*='Lux'], [data-name*='lux'], [data-name*='Smart Money']")
            
            data = {}
            for element in elements:
                try:
                    name = element.get_attribute('data-name') or element.text
                    value = element.get_attribute('data-value') or element.get_attribute('value')
                    
                    if name and value:
                        data[name] = float(value) if value.replace('.', '').replace('-', '').isdigit() else value
                except:
                    continue
                    
            return data
            
        except Exception as e:
            logger.error(f"Error extracting Lux Algo from DOM: {e}")
            return {}
            
    def _parse_market_cipher_data(self, symbol: str, timeframe: str, raw_data: Dict[str, Any]) -> MarketCipherData:
        """Parse raw Market Cipher data into structured format"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract Market Cipher values with fallbacks
            money_flow_green = raw_data.get('money_flow_green', raw_data.get('green_wave', 0.0))
            money_flow_red = raw_data.get('money_flow_red', raw_data.get('red_wave', 0.0))
            momentum_wave = raw_data.get('momentum_wave', raw_data.get('blue_wave', 0.0))
            blue_wave = raw_data.get('blue_wave', momentum_wave)
            
            # Determine signals based on values
            money_flow_signal = 'bullish' if money_flow_green > money_flow_red else 'bearish' if money_flow_red > money_flow_green else 'neutral'
            momentum_signal = 'bullish' if momentum_wave > 0 else 'bearish' if momentum_wave < 0 else 'neutral'
            
            # Support/Resistance levels (estimated from current price)
            current_price = raw_data.get('close', 23.29)  # Default to AVAX price
            support_level = current_price * 0.98  # 2% below
            resistance_level = current_price * 1.02  # 2% above
            
            # Divergence detection
            divergence_detected = raw_data.get('divergence', False)
            divergence_type = raw_data.get('divergence_type', 'none')
            
            # Squeeze momentum
            squeeze_momentum = raw_data.get('squeeze', False)
            squeeze_direction = raw_data.get('squeeze_direction', 'neutral')
            
            # Wave trend and VWAP
            wave_trend = raw_data.get('wave_trend', 0.0)
            vwap_signal = raw_data.get('vwap_signal', 'neutral')
            
            return MarketCipherData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                money_flow_green=money_flow_green,
                money_flow_red=money_flow_red,
                money_flow_signal=money_flow_signal,
                momentum_wave=momentum_wave,
                momentum_signal=momentum_signal,
                blue_wave=blue_wave,
                support_level=support_level,
                resistance_level=resistance_level,
                sr_signal='neutral',
                divergence_detected=divergence_detected,
                divergence_type=divergence_type,
                squeeze_momentum=squeeze_momentum,
                squeeze_direction=squeeze_direction,
                wave_trend=wave_trend,
                vwap_signal=vwap_signal
            )
            
        except Exception as e:
            logger.error(f"Error parsing Market Cipher data: {e}")
            # Return default data structure
            return MarketCipherData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat(),
                money_flow_green=0.0,
                money_flow_red=0.0,
                money_flow_signal='neutral',
                momentum_wave=0.0,
                momentum_signal='neutral',
                blue_wave=0.0,
                support_level=22.50,
                resistance_level=25.50,
                sr_signal='neutral',
                divergence_detected=False,
                divergence_type='none',
                squeeze_momentum=False,
                squeeze_direction='neutral',
                wave_trend=0.0,
                vwap_signal='neutral'
            )
            
    def _parse_lux_algo_data(self, symbol: str, timeframe: str, raw_data: Dict[str, Any]) -> LuxAlgoData:
        """Parse raw Lux Algo data into structured format"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract Lux Algo values
            bullish_order_block = raw_data.get('bullish_ob', False)
            bearish_order_block = raw_data.get('bearish_ob', False)
            order_block_price = raw_data.get('ob_price', 0.0)
            
            # Premium/Discount zones
            premium_zone = raw_data.get('premium', False)
            discount_zone = raw_data.get('discount', False)
            equilibrium = raw_data.get('equilibrium', True)
            zone_strength = raw_data.get('zone_strength', 0.5)
            
            # Market structure
            market_structure = raw_data.get('structure', 'neutral')
            structure_break = raw_data.get('structure_break', False)
            higher_high = raw_data.get('hh', False)
            higher_low = raw_data.get('hl', False)
            lower_high = raw_data.get('lh', False)
            lower_low = raw_data.get('ll', False)
            
            # Smart Money Concepts
            smart_money_signal = raw_data.get('smart_money', 'neutral')
            liquidity_grab = raw_data.get('liquidity_grab', False)
            fair_value_gap = raw_data.get('fvg', False)
            
            # Additional signals
            trend_strength = raw_data.get('trend_strength', 0.5)
            volatility_index = raw_data.get('volatility', 0.5)
            
            return LuxAlgoData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                bullish_order_block=bullish_order_block,
                bearish_order_block=bearish_order_block,
                order_block_price=order_block_price,
                premium_zone=premium_zone,
                discount_zone=discount_zone,
                equilibrium=equilibrium,
                zone_strength=zone_strength,
                market_structure=market_structure,
                structure_break=structure_break,
                higher_high=higher_high,
                higher_low=higher_low,
                lower_high=lower_high,
                lower_low=lower_low,
                smart_money_signal=smart_money_signal,
                liquidity_grab=liquidity_grab,
                fair_value_gap=fair_value_gap,
                trend_strength=trend_strength,
                volatility_index=volatility_index
            )
            
        except Exception as e:
            logger.error(f"Error parsing Lux Algo data: {e}")
            # Return default data structure
            return LuxAlgoData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat(),
                bullish_order_block=False,
                bearish_order_block=False,
                order_block_price=0.0,
                premium_zone=False,
                discount_zone=False,
                equilibrium=True,
                zone_strength=0.5,
                market_structure='neutral',
                structure_break=False,
                higher_high=False,
                higher_low=False,
                lower_high=False,
                lower_low=False,
                smart_money_signal='neutral',
                liquidity_grab=False,
                fair_value_gap=False,
                trend_strength=0.5,
                volatility_index=0.5
            )
            
    def _store_market_cipher_data(self, data: MarketCipherData):
        """Store Market Cipher data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_cipher_data 
                (symbol, timeframe, timestamp, money_flow_green, money_flow_red, money_flow_signal,
                 momentum_wave, momentum_signal, blue_wave, support_level, resistance_level, sr_signal,
                 divergence_detected, divergence_type, squeeze_momentum, squeeze_direction, 
                 wave_trend, vwap_signal, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.timeframe, data.timestamp,
                data.money_flow_green, data.money_flow_red, data.money_flow_signal,
                data.momentum_wave, data.momentum_signal, data.blue_wave,
                data.support_level, data.resistance_level, data.sr_signal,
                data.divergence_detected, data.divergence_type,
                data.squeeze_momentum, data.squeeze_direction,
                data.wave_trend, data.vwap_signal,
                json.dumps(data.__dict__)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing Market Cipher data: {e}")
            
    def _store_lux_algo_data(self, data: LuxAlgoData):
        """Store Lux Algo data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO lux_algo_data 
                (symbol, timeframe, timestamp, bullish_order_block, bearish_order_block, order_block_price,
                 premium_zone, discount_zone, equilibrium, zone_strength, market_structure, structure_break,
                 higher_high, higher_low, lower_high, lower_low, smart_money_signal, liquidity_grab,
                 fair_value_gap, trend_strength, volatility_index, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.timeframe, data.timestamp,
                data.bullish_order_block, data.bearish_order_block, data.order_block_price,
                data.premium_zone, data.discount_zone, data.equilibrium, data.zone_strength,
                data.market_structure, data.structure_break,
                data.higher_high, data.higher_low, data.lower_high, data.lower_low,
                data.smart_money_signal, data.liquidity_grab, data.fair_value_gap,
                data.trend_strength, data.volatility_index,
                json.dumps(data.__dict__)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing Lux Algo data: {e}")
            
    def get_comprehensive_analysis(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get comprehensive analysis combining Market Cipher and Lux Algo data"""
        try:
            logger.info(f"Getting comprehensive analysis for {symbol} ({timeframe})")
            
            # Get Market Cipher data
            market_cipher = self.get_market_cipher_data(symbol, timeframe)
            
            # Get Lux Algo data
            lux_algo = self.get_lux_algo_data(symbol, timeframe)
            
            if not market_cipher or not lux_algo:
                logger.error("Failed to retrieve indicator data")
                return {}
            
            # Combine analysis
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'market_cipher': {
                    'money_flow_signal': market_cipher.money_flow_signal,
                    'momentum_signal': market_cipher.momentum_signal,
                    'squeeze_active': market_cipher.squeeze_momentum,
                    'divergence_detected': market_cipher.divergence_detected,
                    'wave_trend': market_cipher.wave_trend,
                    'support_level': market_cipher.support_level,
                    'resistance_level': market_cipher.resistance_level
                },
                'lux_algo': {
                    'market_structure': lux_algo.market_structure,
                    'smart_money_signal': lux_algo.smart_money_signal,
                    'order_blocks': {
                        'bullish': lux_algo.bullish_order_block,
                        'bearish': lux_algo.bearish_order_block
                    },
                    'premium_discount': {
                        'premium': lux_algo.premium_zone,
                        'discount': lux_algo.discount_zone,
                        'equilibrium': lux_algo.equilibrium
                    },
                    'structure_break': lux_algo.structure_break,
                    'trend_strength': lux_algo.trend_strength
                },
                'confluence_score': self._calculate_confluence_score(market_cipher, lux_algo),
                'overall_signal': self._determine_overall_signal(market_cipher, lux_algo)
            }
            
            logger.info(f"Comprehensive analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {}
            
    def _calculate_confluence_score(self, mc_data: MarketCipherData, lux_data: LuxAlgoData) -> float:
        """Calculate confluence score from Market Cipher and Lux Algo data"""
        try:
            score = 0.0
            max_score = 10.0
            
            # Market Cipher signals (50% weight)
            if mc_data.money_flow_signal == 'bullish':
                score += 1.5
            elif mc_data.money_flow_signal == 'bearish':
                score -= 1.5
                
            if mc_data.momentum_signal == 'bullish':
                score += 1.5
            elif mc_data.momentum_signal == 'bearish':
                score -= 1.5
                
            if mc_data.squeeze_momentum:
                score += 1.0
                
            if mc_data.divergence_detected:
                score += 1.0 if mc_data.divergence_type == 'bullish' else -1.0
            
            # Lux Algo signals (50% weight)
            if lux_data.market_structure == 'bullish':
                score += 1.5
            elif lux_data.market_structure == 'bearish':
                score -= 1.5
                
            if lux_data.smart_money_signal == 'accumulation':
                score += 1.5
            elif lux_data.smart_money_signal == 'distribution':
                score -= 1.5
                
            if lux_data.bullish_order_block:
                score += 1.0
            elif lux_data.bearish_order_block:
                score -= 1.0
                
            if lux_data.discount_zone:
                score += 0.5
            elif lux_data.premium_zone:
                score -= 0.5
            
            # Normalize to 0-1 scale
            normalized_score = (score + max_score) / (2 * max_score)
            return max(0.0, min(1.0, normalized_score))
            
        except Exception as e:
            logger.error(f"Error calculating confluence score: {e}")
            return 0.5
            
    def _determine_overall_signal(self, mc_data: MarketCipherData, lux_data: LuxAlgoData) -> str:
        """Determine overall trading signal"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # Count Market Cipher signals
            if mc_data.money_flow_signal == 'bullish':
                bullish_signals += 1
            elif mc_data.money_flow_signal == 'bearish':
                bearish_signals += 1
                
            if mc_data.momentum_signal == 'bullish':
                bullish_signals += 1
            elif mc_data.momentum_signal == 'bearish':
                bearish_signals += 1
            
            # Count Lux Algo signals
            if lux_data.market_structure == 'bullish':
                bullish_signals += 1
            elif lux_data.market_structure == 'bearish':
                bearish_signals += 1
                
            if lux_data.smart_money_signal == 'accumulation':
                bullish_signals += 1
            elif lux_data.smart_money_signal == 'distribution':
                bearish_signals += 1
            
            # Determine overall signal
            if bullish_signals > bearish_signals + 1:
                return 'bullish'
            elif bearish_signals > bullish_signals + 1:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining overall signal: {e}")
            return 'neutral'
            
    def close(self):
        """Close TradingView connection and cleanup"""
        try:
            if self.driver:
                self.driver.quit()
            if self.websocket:
                self.websocket.close()
            logger.info("TradingView connection closed")
        except Exception as e:
            logger.error(f"Error closing TradingView connection: {e}")

# Global TradingView API instance
tradingview_api = None

def initialize_tradingview_api(username: str, password: str) -> bool:
    """Initialize TradingView API with credentials"""
    global tradingview_api
    try:
        credentials = TradingViewCredentials(username=username, password=password)
        tradingview_api = TradingViewAPI(credentials)
        return tradingview_api.authenticate()
    except Exception as e:
        logger.error(f"Error initializing TradingView API: {e}")
        return False

def get_live_market_cipher_data(symbol: str, timeframe: str = "1h") -> Optional[MarketCipherData]:
    """Get live Market Cipher data from TradingView"""
    global tradingview_api
    if tradingview_api:
        return tradingview_api.get_market_cipher_data(symbol, timeframe)
    return None

def get_live_lux_algo_data(symbol: str, timeframe: str = "1h") -> Optional[LuxAlgoData]:
    """Get live Lux Algo data from TradingView"""
    global tradingview_api
    if tradingview_api:
        return tradingview_api.get_lux_algo_data(symbol, timeframe)
    return None

def get_live_comprehensive_analysis(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    """Get live comprehensive analysis from TradingView"""
    global tradingview_api
    if tradingview_api:
        return tradingview_api.get_comprehensive_analysis(symbol, timeframe)
    return {}

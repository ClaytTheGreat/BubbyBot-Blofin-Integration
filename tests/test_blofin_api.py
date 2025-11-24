"""
Unit tests for Blofin API Client
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blofin.api_client import BlofinAPIClient
from src.blofin.exchange_adapter import BlofinExchangeAdapter, TradingSignal
from config.blofin_config import get_config
from datetime import datetime


class TestBlofinAPIClient:
    """Test Blofin API Client"""
    
    def test_signature_generation(self):
        """Test signature generation"""
        client = BlofinAPIClient(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_pass",
            demo=True
        )
        
        # Test signature generation
        signature, timestamp, nonce = client._generate_signature(
            method="GET",
            path="/api/v1/account/balance"
        )
        
        assert signature is not None
        assert len(signature) > 0
        assert timestamp is not None
        assert nonce is not None
    
    def test_signature_with_body(self):
        """Test signature generation with request body"""
        client = BlofinAPIClient(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_pass",
            demo=True
        )
        
        body = {
            "instId": "BTC-USDT",
            "side": "buy",
            "size": "1"
        }
        
        signature, timestamp, nonce = client._generate_signature(
            method="POST",
            path="/api/v1/trade/order",
            body=body
        )
        
        assert signature is not None
        assert len(signature) > 0


class TestBlofinExchangeAdapter:
    """Test Blofin Exchange Adapter"""
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        config = get_config(demo=True)
        
        # Mock API client
        class MockAPIClient:
            pass
        
        adapter = BlofinExchangeAdapter(MockAPIClient(), config)
        
        signal = TradingSignal(
            instrument="BTC-USDT",
            side="buy",
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=53000.0,
            timeframe="15m",
            pattern_type="test",
            timestamp=datetime.now()
        )
        
        position_size = adapter.calculate_position_size(
            signal=signal,
            account_balance=10000.0
        )
        
        assert position_size > 0
        assert position_size >= 0.1  # Minimum size
    
    def test_leverage_calculation(self):
        """Test leverage calculation based on confidence"""
        config = get_config(demo=True)
        
        class MockAPIClient:
            pass
        
        adapter = BlofinExchangeAdapter(MockAPIClient(), config)
        
        # High confidence
        leverage_high = adapter._calculate_leverage(0.95)
        assert leverage_high == 50
        
        # Medium confidence
        leverage_med = adapter._calculate_leverage(0.85)
        assert leverage_med == 35
        
        # Low confidence
        leverage_low = adapter._calculate_leverage(0.75)
        assert leverage_low == 20


class TestConfiguration:
    """Test configuration"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = get_config(demo=True)
        
        assert config is not None
        assert config['demo']['enabled'] == True
        assert config['risk_management']['mandatory_stop_loss'] == True
        assert config['trading']['max_leverage'] <= 150
    
    def test_risk_management_config(self):
        """Test risk management configuration"""
        config = get_config(demo=True)
        
        assert config['risk_management']['max_position_size_pct'] <= 0.10
        assert config['risk_management']['min_risk_reward_ratio'] >= 1.0
        assert config['risk_management']['mandatory_stop_loss'] == True


class TestTradingSignal:
    """Test TradingSignal dataclass"""
    
    def test_signal_creation(self):
        """Test creating a trading signal"""
        signal = TradingSignal(
            instrument="BTC-USDT",
            side="buy",
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=53000.0,
            timeframe="15m",
            pattern_type="divergence",
            timestamp=datetime.now()
        )
        
        assert signal.instrument == "BTC-USDT"
        assert signal.side == "buy"
        assert signal.confidence == 0.85
        assert signal.stop_loss < signal.entry_price  # For buy signal
        assert signal.take_profit > signal.entry_price  # For buy signal
    
    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation"""
        signal = TradingSignal(
            instrument="BTC-USDT",
            side="buy",
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=49000.0,  # 2% risk
            take_profit=53000.0,  # 6% reward
            timeframe="15m",
            pattern_type="divergence",
            timestamp=datetime.now()
        )
        
        risk = signal.entry_price - signal.stop_loss
        reward = signal.take_profit - signal.entry_price
        risk_reward_ratio = reward / risk
        
        assert risk_reward_ratio >= 2.0  # Minimum 2:1 ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

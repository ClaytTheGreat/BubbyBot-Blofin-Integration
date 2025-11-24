# Blofin Exchange API Research

## Overview

**Blofin** is a cryptocurrency exchange specializing in futures and perpetual contract trading. It offers:
- Over 400 USDT-M trading pairs
- Leverage up to 150x
- Spot and futures trading
- Copy trading features
- Trading bots
- Demo trading environment

## API Information

### Base URLs
- **Production REST API**: `https://openapi.blofin.com`
- **Public WebSocket**: `wss://openapi.blofin.com/ws/public`
- **Private WebSocket**: `wss://openapi.blofin.com/ws/private`
- **Demo Trading REST API**: `https://demo-trading-openapi.blofin.com`
- **Demo Public WebSocket**: `wss://demo-trading-openapi.blofin.com/ws/public`
- **Demo Private WebSocket**: `wss://demo-trading-openapi.blofin.com/ws/private`

### API Features
- **Efficient order management**
- **Market data access**
- **Position monitoring**
- **Risk management tools**
- **Account operations**
- **Copy Trading support**

### Authentication

All private REST requests require the following headers:
- `ACCESS-KEY`: The API Key as a String
- `ACCESS-SIGN`: The Base64-encoded signature
- `ACCESS-TIMESTAMP`: UTC timestamp in milliseconds (e.g., 1597026383085)
- `ACCESS-NONCE`: Unique identifier (UUID, Snowflake algorithm, etc.)
- `ACCESS-PASSPHRASE`: Passphrase specified during API Key creation

### API Permissions
- **READ**: View account info, bills, order history
- **TRADE**: Place and cancel orders, view account info
- **TRANSFER**: Make funding transfers between accounts

### Signature Generation Process

1. Create prehash string by concatenating:
   - `requestPath` (including query parameters for GET requests)
   - `method` (HTTP method in uppercase: GET, POST, etc.)
   - `timestamp` (milliseconds since epoch)
   - `nonce` (unique identifier like UUID)
   - `body` (JSON string for POST requests, empty string for GET)

2. Generate HMAC-SHA256 signature using the SecretKey
3. Convert signature to hexadecimal
4. Encode the hex signature in Base64 format

### Python Signature Example

```python
import hmac
import hashlib
import base64
import json
from datetime import datetime
from uuid import uuid4

def sign_request(secret: str, method: str, path: str, body: dict | None = None) -> tuple:
    """Generate BloFin API request signature.
    
    Returns:
        tuple: (signature, timestamp, nonce)
    """
    timestamp = str(int(datetime.now().timestamp() * 1000))
    nonce = str(uuid4())
    
    # Create prehash string
    msg = f"{path}{method}{timestamp}{nonce}"
    if body:
        msg += json.dumps(body)
        
    # Generate hex signature and convert to base64
    hex_signature = hmac.new(
        secret.encode(),
        msg.encode(),
        hashlib.sha256
    ).hexdigest().encode()
    
    signature = base64.b64encode(hex_signature).decode()
    
    return signature, timestamp, nonce
```

### Example API Endpoints

#### GET Request Example
```python
path = "/api/v1/asset/balances?accountType=futures"
method = "GET"
body = ""  # Empty for GET requests
```

#### POST Request Example (Place Order)
```python
path = "/api/v1/trade/order"
method = "POST"
body = {
    "instId": "BTC-USDT",
    "marginMode": "isolated",
    "side": "buy",
    "orderType": "limit",
    "price": "35000",
    "size": "0.1"  # Minimum order size is 0.1 contracts
}
```

## API Sections

Based on the documentation, Blofin API includes:

1. **General Info** - Basic API information
2. **API Key Creation** - How to create and manage API keys
3. **REST Authentication** - Authentication mechanism
4. **WebSocket** - Real-time data streaming
5. **Rate Limits** - API rate limiting information
6. **Risk Control Restrictions** - Trading restrictions
7. **Public Data** - Market data endpoints
8. **Account** - Account management endpoints
9. **Trading** - Order placement and management
10. **Affiliate** - Affiliate program endpoints
11. **Copy Trading** - Copy trading functionality
12. **User** - User management
13. **Errors** - Error codes and handling

## Key Notes

- All timestamps are in **milliseconds**
- Data is returned in **descending order** (newest first, oldest last)
- All endpoints return JSON object or array
- Request bodies must have content type `application/json`
- API Keys can be linked with up to 20 IP addresses
- API Keys not bound to IPs expire after 90 days

## Integration Requirements for BubbyBot

To integrate BubbyBot with Blofin, we need to:

1. **Create a Blofin API client** with proper authentication
2. **Implement order placement functions** (market, limit orders)
3. **Implement position management** (get positions, close positions)
4. **Implement TP/SL management** (set take profit and stop loss)
5. **Implement account balance retrieval**
6. **Handle WebSocket connections** for real-time data
7. **Implement error handling** for API responses
8. **Add rate limiting** to comply with API restrictions

## Documentation URL

Full API documentation: https://docs.blofin.com/index.html


## Detailed Trading API Endpoints

### 1. Place Order
**Endpoint**: `POST /api/v1/trade/order`

**Request Parameters**:
- `instId` (String, Required): Instrument ID, e.g., BTC-USDT
- `marginMode` (String, Required): Margin mode - `cross` or `isolated`
- `positionSide` (String, Required): Position side
  - Default `net` for One-way Mode
  - `long` or `short` for Hedge Mode
- `side` (String, Required): Order side - `buy` or `sell`
- `orderType` (String, Required): Order type
  - `market`: market order
  - `limit`: limit order
  - `post_only`: Post-only order
  - `fok`: Fill-or-kill order
  - `ioc`: Immediate-or-cancel order
- `size` (String, Yes): Number of contracts to buy or sell
- `price` (String, No): Order price (required for limit orders)
- `reduceOnly` (String, No): Whether orders can only reduce position size
  - Valid options: `true` or `false`. Default is `false`
  - When `reduceOnly = true` and opposite order size exceeds position size, position will be fully closed
- `clientOrderId` (String, No): Client Order ID (up to 32 characters)
- **`tpTriggerPrice`** (String, No): **Take-profit trigger price**
  - If you fill in this parameter, you should fill in the `tpOrderPrice` as well
- **`tpOrderPrice`** (String, No): **Take-profit order price**
  - If you fill in this parameter, you should fill in the `tpTriggerPrice` as well
  - If the price is -1, take-profit will be executed at the market price
- **`slTriggerPrice`** (String, No): **Stop-loss trigger price**
  - If you fill in this parameter, you should fill in the `slOrderPrice` as well
- **`slOrderPrice`** (String, No): **Stop-loss order price**
  - If you fill in this parameter, you should fill in the `slTriggerPrice` as well
  - If the price is -1, stop-loss will be executed at the market price
- `brokerId` (String, No): Broker ID provided by BloFin (up to 16 characters)

**Example Request**:
```python
{
    "instId": "BTC-USDT",
    "marginMode": "cross",
    "positionSide": "long",
    "side": "sell",
    "price": "23212.2",
    "size": "2"
}
```

**Example with TP/SL**:
```python
{
    "instId": "BTC-USDT",
    "marginMode": "isolated",
    "positionSide": "long",
    "side": "buy",
    "orderType": "market",
    "size": "1",
    "tpTriggerPrice": "50000",
    "tpOrderPrice": "50000",
    "slTriggerPrice": "30000",
    "slOrderPrice": "-1"  # Market price for stop loss
}
```

### 2. Place TPSL Order
**Endpoint**: `POST /api/v1/trade/order-tpsl`

Used to place standalone Take Profit / Stop Loss orders for existing positions.

### 3. GET Positions
**Endpoint**: `GET /api/v1/account/positions`

Retrieve current open positions.

### 4. Close Positions
**Endpoint**: `POST /api/v1/trade/close-positions`

Close all or specific positions.

### 5. Cancel Order
**Endpoint**: `POST /api/v1/trade/cancel-order`

Cancel a specific order.

### 6. GET Active Orders
**Endpoint**: `GET /api/v1/trade/orders-pending`

Get all active (pending) orders.

### 7. GET Order History
**Endpoint**: `GET /api/v1/trade/orders-history`

Get historical orders.

### 8. Set Leverage
**Endpoint**: `POST /api/v1/account/set-leverage`

Set leverage for a specific instrument.

## Key Features for BubbyBot Integration

1. **Simultaneous TP/SL on Order Placement**: Blofin allows setting TP and SL directly when placing an order using `tpTriggerPrice`, `tpOrderPrice`, `slTriggerPrice`, and `slOrderPrice` parameters.

2. **Market Price TP/SL**: Setting order price to `-1` executes TP/SL at market price.

3. **Reduce-Only Orders**: Can set `reduceOnly=true` to ensure orders only close positions.

4. **Position Management**: Supports both One-way Mode (`net`) and Hedge Mode (`long`/`short`).

5. **Margin Modes**: Supports both `cross` and `isolated` margin modes.

6. **Order Types**: Supports market, limit, post-only, FOK, and IOC orders.

## WebSocket Support

Blofin provides WebSocket APIs for:
- Real-time market data
- Real-time position updates
- Real-time order updates
- Private account data

WebSocket endpoints:
- Public: `wss://openapi.blofin.com/ws/public`
- Private: `wss://openapi.blofin.com/ws/private`

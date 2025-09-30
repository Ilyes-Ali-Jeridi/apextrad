from decimal import Decimal

class Config:
    # Trading pair settings
    TRADING_PAIR = "XRPUSDT"
    BASE_CURRENCY = "XRP"
    QUOTE_CURRENCY = "USDT"

    # Account and balance settings
    INITIAL_BALANCE = 50.0  # Initial balance in quote currency for paper trading
    PAPER_TRADING = True    # Set to True to simulate trades, False for live trading

    # Model and training settings
    BATCH_SIZE = 256  # Number of samples per training batch
    UNCERTAINTY_THRESHOLD = 0.15  # Threshold for model uncertainty; trades are only executed if uncertainty is below this value
    QUANTUM_ADJUSTMENT_FACTOR = 0.003  # Factor for adjusting order price based on quantum circuit output

    # Risk management settings
    RISK_CAP = 0.01  # Maximum percentage of total balance to risk on a single trade
    BASE_POSITION = 0.01  # Base position size as a percentage of the total balance
    RISK_LEVERAGE = 0.1  # Leverage to apply to the PPO agent's action for position sizing
    MAX_POSITION = 0.1  # Maximum allowed position size as a percentage of the total balance
    TARGET_DAILY = 0.02  # Target daily return
    QUANTUM_SPREAD_THRESHOLD = 0.001 # Threshold for spread to be considered in quantum adjustment

    # Precision settings for orders
    QTY_PRECISION = Decimal('0.01')  # Precision for the quantity of the base currency
    PRICE_PRECISION = Decimal('0.00001')  # Precision for the price of the trading pair
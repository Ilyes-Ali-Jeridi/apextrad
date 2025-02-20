from decimal import Decimal

class Config:
    TRADING_PAIR = "XRPUSDT"
    BASE_CURRENCY = "XRP"
    QUOTE_CURRENCY = "USDT"
    INITIAL_BALANCE = 50.0
    BATCH_SIZE = 256
    QTY_PRECISION = Decimal('0.01')
    PRICE_PRECISION = Decimal('0.00001')
    UNCERTAINTY_THRESHOLD = 0.15
    QUANTUM_ADJUSTMENT_FACTOR = 0.003
    RISK_CAP = 0.01       # Reduced risk cap for paper trading - VERY IMPORTANT
    BASE_POSITION = 0.01  # Reduced base position for paper trading - VERY IMPORTANT
    RISK_LEVERAGE = 0.1
    MAX_POSITION = 0.1
    TARGET_DAILY = 0.02
    QUANTUM_SPREAD_THRESHOLD = 0.001
    PAPER_TRADING = True # Enable Paper Trading Mode - NEW CONFIGURATION
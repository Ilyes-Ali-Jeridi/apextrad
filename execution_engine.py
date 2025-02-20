import asyncio
import logging
import time
import numpy as np
import cirq
import tensorflow_quantum as tfq
from decimal import Decimal
from binance.client import AsyncClient
from binance.streams import BinanceSocketManager
from collections import deque
from .config import Config # Import Config from config.py

class QuantumExecutionEngine:
    """Quantum-Informed Order Routing System - PAPER TRADING VERSION"""
    RETRY_DELAY_BASE = 1
    MAX_RETRIES = 3

    def __init__(self, api_key: str, api_secret: str):
        self.client = AsyncClient(api_key, api_secret)
        self.ws_manager = BinanceSocketManager(self.client)
        self.position = Decimal('0.0')
        self.balance = Decimal(Config.INITIAL_BALANCE)
        self.order_book = {'bids': [], 'asks': []}

        # Quantum state tracking
        self.qubit_state = np.zeros(4)
        self.quantum_risk = 1.0
        self.quantum_circuit = self._build_quantum_execution_circuit()

        # Initialize feature cache
        self.feature_cache = deque(maxlen=390)

    async def initialize(self):
        await self._update_account_state()
        await self._init_order_book()

    async def _update_account_state(self):
        if Config.PAPER_TRADING:
            logging.info(f"Paper Trading Mode: Initializing balance to {Config.INITIAL_BALANCE} {Config.BASE_CURRENCY}")
            print(f"Paper Trading Mode: Initializing balance to {Config.INITIAL_BALANCE} {Config.BASE_CURRENCY}")
            return
        try:
            account = await self.client.get_account()
            self.balance = Decimal(account['totalWalletBalance'])
            logging.info(f"Account balance updated to: {self.balance} {Config.BASE_CURRENCY} (Live Account)")
            print(f"Account balance updated to: {self.balance} {Config.BASE_CURRENCY} (Live Account)")
        except Exception as e:
            logging.error(f"Error updating account balance: {e}")
            print(f"Error updating account balance: {e}")
            self.balance = Decimal(Config.INITIAL_BALANCE)

    async def _init_order_book(self):
        try:
            depth = await self.client.get_order_book(symbol=Config.TRADING_PAIR)
            self.order_book = {
                'bids': sorted(depth['bids'], key=lambda x: float(x[0]), reverse=True),
                'asks': sorted(depth['asks'], key=lambda x: float(x[0]))
            }
            logging.info(f"Order book initialized for {Config.TRADING_PAIR}")
            print(f"Order book initialized for {Config.TRADING_PAIR}")
        except Exception as e:
            logging.error(f"Error initializing order book: {e}")
            print(f"Error initializing order book: {e}")
            self.order_book = {'bids': [], 'asks': []}

    async def execute_order(self, side: str, qty: float, limit_price: float):
        """Quantum-optimized order execution - PAPER TRADING SIMULATION"""
        if Config.PAPER_TRADING:
            logging.info(f"PAPER TRADE: Simulating {side} order for {qty} {Config.TRADING_PAIR} at {limit_price}")
            print(f"PAPER TRADE: Simulating {side} order for {qty} {Config.TRADING_PAIR} at {limit_price}")
            fill_price = limit_price
            fill_qty = qty
            cost = fill_price * fill_qty

            if side == 'BUY':
                self.position += fill_qty
                self.balance -= cost
            elif side == 'SELL':
                self.position -= fill_qty
                self.balance += cost

            logging.info(f"PAPER TRADE: Order simulated - Filled {fill_qty} at {fill_price}. New Position: {self.position}, New Balance: {self.balance}")
            print(f"PAPER TRADE: Order simulated - Filled {fill_qty} at {fill_price}. New Position: {self.position}, New Balance: {self.balance}")
            return True

        for retry_attempt in range(self.MAX_RETRIES):
            try:
                if not self.order_book['bids'] or not self.order_book['asks']:
                    logging.warning("Order book empty, cannot execute order.")
                    return False

                best_bid_price = Decimal(self.order_book['bids'][0][0])
                best_ask_price = Decimal(self.order_book['asks'][0][0])
                self.best_bid = best_bid_price
                self.best_ask = best_ask_price
                mid_price = (self.best_bid + self.best_ask) / 2
                quantum_adjustment = self._calculate_quantum_adjustment()

                if side == 'BUY':
                    price = mid_price * (1 - quantum_adjustment)
                else:
                    price = mid_price * (1 + quantum_adjustment)

                price_quantized = Decimal(str(price)).quantize(Config.PRICE_PRECISION)
                qty_sized = self._size_position(qty)
                order = await self.client.create_order(
                    symbol=Config.TRADING_PAIR,
                    side=side,
                    type='LIMIT',
                    timeInForce='IOC',
                    quantity=str(qty_sized),
                    price=str(price_quantized)
                )

                logging.info(f"Order placed: {side} {qty_sized} {Config.TRADING_PAIR} at {price_quantized}")
                await self._process_order_fills(order)
                return True

            except Exception as e:
                retry_delay = self.RETRY_DELAY_BASE * (2 ** retry_attempt)
                logging.error(f"Execution error (attempt {retry_attempt + 1}/{self.MAX_RETRIES}): {e}. Retrying in {retry_delay} seconds...")
                if retry_attempt == self.MAX_RETRIES - 1:
                    logging.error(f"Max retries reached for order execution. Order failed.")
                    return False
                await asyncio.sleep(retry_delay)

    def _calculate_quantum_adjustment(self):
        """Quantum state-informed price adjustment"""
        try:
            spread = (self.best_ask - self.best_bid) if hasattr(self, 'best_bid') and hasattr(self, 'best_ask') and self.best_bid and self.best_ask else 0.0
            if spread < 0:
                spread = 0.0

            scaled_spread = min(1.0, max(0.0, float(spread) / Config.QUANTUM_SPREAD_THRESHOLD))

            adjustment = tfq.layers.Expectation()(
                self.quantum_circuit,
                symbol_names=['qubit_state', 'spread_param'],
                symbol_values=[self.qubit_state, scaled_spread]
            ).numpy()[0] * Config.QUANTUM_ADJUSTMENT_FACTOR
            logging.debug(f"Quantum adjustment calculated: {adjustment} based on spread: {spread}")
            return adjustment
        except Exception as e:
            logging.error(f"Quantum Adjustment Error: {e}. Returning 0 adjustment.")
            return 0.0

    def _size_position(self, qty: float):
        """Risk-managed position sizing"""
        max_size_usd = self.balance * Decimal(Config.RISK_CAP)
        current_price = (self.best_bid + self.best_ask) / 2 if hasattr(self, 'best_bid') and hasattr(self, 'best_ask') else 1.0
        max_qty_base_currency = max_size_usd / current_price
        qty_to_trade = min(qty, max_qty_base_currency)
        sized_qty = qty_to_trade.quantize(Config.QTY_PRECISION)
        logging.debug(f"Position sized to: {sized_qty} {Config.BASE_CURRENCY}")
        return sized_qty

    def _build_quantum_execution_circuit(self):
        """Quantum circuit for execution adjustment"""
        circuit = cirq.Circuit()
        q = cirq.GridQubit(0, 0)
        spread_param = cirq.ParamResolver('spread_param')
        circuit.append(cirq.rx(np.pi * spread_param).on(q))
        logging.debug("Quantum execution circuit built (spread-dependent)")
        return circuit

    async def _process_order_fills(self, order_response):
        """Process order fill information and update position/balance - IMPROVED"""
        if Config.PAPER_TRADING:
            logging.info("PAPER TRADE: Order fill processing skipped.")
            print("PAPER TRADE: Order fill processing skipped.")
            return

        try:
            if order_response['status'] == 'FILLED':
                fills = order_response.get('fills', [])

                if fills:
                    filled_qty = Decimal('0.0')
                    cost = Decimal('0.0')
                    for fill in fills:
                        fill_price = Decimal(fill['price'])
                        fill_qty = Decimal(fill['qty'])
                        commission = Decimal(fill['commission'])
                        commission_asset = fill['commissionAsset']

                        filled_qty += fill_qty
                        cost += fill_price * fill_qty

                    trade_side = order_response['side']
                    if trade_side == 'BUY':
                        self.position += filled_qty
                        self.balance -= cost
                    elif trade_side == 'SELL':
                        self.position -= fill_qty
                        self.balance += cost

                    logging.info(f"Order FILLED: {trade_side} {filled_qty} {Config.TRADING_PAIR} at avg price {cost/filled_qty if filled_qty else 0}. New Position: {self.position} {Config.BASE_CURRENCY}, New Balance: {self.balance} {Config.QUOTE_CURRENCY}")
                    print(f"PAPER TRADE: Order simulated - Filled {fill_qty} at avg price {cost/filled_qty if filled_qty else 0}. New Position: {self.position} {Config.BASE_CURRENCY}, New Balance: {self.balance} {Config.QUOTE_CURRENCY}")

                else:
                    logging.warning(f"Order FILLED but no fills found in response: {order_response}")
                    print(f"Order FILLED but no fills found in response: {order_response}")

            else:
                logging.info(f"Order status: {order_response['status']} for order: {order_response['orderId']}")
                print(f"Order status: {order_response['status']} for order: {order_response['orderId']}")

        except Exception as e:
            logging.error(f"Error processing order fill: {e}. Order response: {order_response}")
            print(f"Error processing order fill: {e}. Order response: {order_response}")
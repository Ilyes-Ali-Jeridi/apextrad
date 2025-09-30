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
from src.config import Config

class QuantumExecutionEngine:
    """
    Handles order execution, risk management, and communication with the
    Binance API. This class can operate in both paper trading and live modes.
    It uses a quantum circuit to make fine-tuned adjustments to order prices.
    """
    RETRY_DELAY_BASE = 1
    MAX_RETRIES = 3

    def __init__(self, api_key: str, api_secret: str):
        """
        Initializes the QuantumExecutionEngine.

        Args:
            api_key (str): The Binance API key.
            api_secret (str): The Binance API secret.
        """
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
        """
        Initializes the execution engine by updating the account state and
        fetching the initial order book.
        """
        await self._update_account_state()
        await self._update_order_book()

    async def _update_account_state(self):
        """
        Updates the account balance. In paper trading mode, it uses the initial
        balance from the config. In live mode, it fetches the balance from Binance.
        """
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

    async def _update_order_book(self):
        """
        Fetches the latest order book by fetching the current depth from Binance.
        """
        try:
            depth = await self.client.get_order_book(symbol=Config.TRADING_PAIR)
            self.order_book = {
                'bids': sorted(depth['bids'], key=lambda x: float(x[0]), reverse=True),
                'asks': sorted(depth['asks'], key=lambda x: float(x[0]))
            }
            logging.info(f"Order book updated for {Config.TRADING_PAIR}")
            print(f"Order book updated for {Config.TRADING_PAIR}")
        except Exception as e:
            logging.error(f"Error updating order book: {e}")
            print(f"Error updating order book: {e}")
            self.order_book = {'bids': [], 'asks': []}

    async def execute_order(self, side: str, qty: float, limit_price: float):
        """
        Executes a trade, either simulated or live.

        In paper trading mode, it simulates the order and updates the balance
        and position. In live mode, it places a limit order on Binance with a
        price adjusted by a quantum circuit.

        Args:
            side (str): The order side ('BUY' or 'SELL').
            qty (float): The quantity to trade.
            limit_price (float): The limit price for the order.

        Returns:
            bool: True if the order was successfully executed or simulated,
                  False otherwise.
        """
        if Config.PAPER_TRADING:
            # Simulate the trade in paper trading mode
            logging.info(f"PAPER TRADE: Simulating {side} order for {qty} {Config.TRADING_PAIR} at {limit_price}")
            print(f"PAPER TRADE: Simulating {side} order for {qty} {Config.TRADING_PAIR} at {limit_price}")
            fill_price = limit_price
            fill_qty = qty
            cost = fill_price * fill_qty

            if side == 'BUY':
                self.position += Decimal(fill_qty)
                self.balance -= Decimal(cost)
            elif side == 'SELL':
                self.position -= Decimal(fill_qty)
                self.balance += Decimal(cost)

            logging.info(f"PAPER TRADE: Order simulated - Filled {fill_qty} at {fill_price}. New Position: {self.position}, New Balance: {self.balance}")
            print(f"PAPER TRADE: Order simulated - Filled {fill_qty} at {fill_price}. New Position: {self.position}, New Balance: {self.balance}")
            return True

        # Execute the trade in live mode
        for retry_attempt in range(self.MAX_RETRIES):
            try:
                await self._update_order_book() # Fetch the latest order book
                if not self.order_book['bids'] or not self.order_book['asks']:
                    logging.warning("Order book empty, cannot execute order.")
                    return False

                # Calculate the mid-price and quantum adjustment
                best_bid_price = Decimal(self.order_book['bids'][0][0])
                best_ask_price = Decimal(self.order_book['asks'][0][0])
                self.best_bid = best_bid_price
                self.best_ask = best_ask_price
                mid_price = (self.best_bid + self.best_ask) / 2
                quantum_adjustment = self._calculate_quantum_adjustment()

                # Adjust the price based on the quantum calculation
                if side == 'BUY':
                    price = mid_price * (1 - quantum_adjustment)
                else:
                    price = mid_price * (1 + quantum_adjustment)

                # Quantize the price and size the position
                price_quantized = Decimal(str(price)).quantize(Config.PRICE_PRECISION)
                qty_sized = self._size_position(qty)

                # Create the order
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
                # Retry with exponential backoff if the order fails
                retry_delay = self.RETRY_DELAY_BASE * (2 ** retry_attempt)
                logging.error(f"Execution error (attempt {retry_attempt + 1}/{self.MAX_RETRIES}): {e}. Retrying in {retry_delay} seconds...")
                if retry_attempt == self.MAX_RETRIES - 1:
                    logging.error(f"Max retries reached for order execution. Order failed.")
                    return False
                await asyncio.sleep(retry_delay)
        return False


    def _calculate_quantum_adjustment(self) -> float:
        """
        Calculates a price adjustment using a parameterized quantum circuit.
        The adjustment is based on the current bid-ask spread.

        Returns:
            float: The calculated price adjustment.
        """
        try:
            spread = (self.best_ask - self.best_bid) if hasattr(self, 'best_bid') and hasattr(self, 'best_ask') and self.best_bid and self.best_ask else 0.0
            if spread < 0:
                spread = 0.0

            # Scale the spread to be used as a parameter in the quantum circuit
            scaled_spread = min(1.0, max(0.0, float(spread) / Config.QUANTUM_SPREAD_THRESHOLD))

            # Calculate the expectation of the quantum circuit
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

    def _size_position(self, qty: float) -> Decimal:
        """
        Sizes the position based on risk management parameters.

        Args:
            qty (float): The initial desired quantity.

        Returns:
            Decimal: The risk-adjusted and quantized quantity.
        """
        max_size_usd = self.balance * Decimal(Config.RISK_CAP)
        current_price = (self.best_bid + self.best_ask) / 2 if hasattr(self, 'best_bid') and hasattr(self, 'best_ask') else Decimal('1.0')
        if current_price == 0:
            return Decimal('0.0')
        max_qty_base_currency = max_size_usd / current_price
        qty_to_trade = min(Decimal(qty), max_qty_base_currency)
        sized_qty = qty_to_trade.quantize(Config.QTY_PRECISION)
        logging.debug(f"Position sized to: {sized_qty} {Config.BASE_CURRENCY}")
        return sized_qty

    def _build_quantum_execution_circuit(self) -> cirq.Circuit:
        """
        Builds the quantum circuit used for price adjustments.

        Returns:
            cirq.Circuit: The constructed quantum circuit.
        """
        circuit = cirq.Circuit()
        q = cirq.GridQubit(0, 0)
        spread_param = cirq.ParamResolver('spread_param')
        circuit.append(cirq.rx(np.pi * spread_param).on(q))
        logging.debug("Quantum execution circuit built (spread-dependent)")
        return circuit

    async def _process_order_fills(self, order_response: dict):
        """
        Processes the fills from an order response and updates the account's
        position and balance accordingly.

        Args:
            order_response (dict): The order response from the Binance API.
        """
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
                        fill_qty_i = Decimal(fill['qty'])
                        commission = Decimal(fill['commission'])
                        commission_asset = fill['commissionAsset']

                        filled_qty += fill_qty_i
                        cost += fill_price * fill_qty_i

                    trade_side = order_response['side']
                    if trade_side == 'BUY':
                        self.position += filled_qty
                        self.balance -= cost
                    elif trade_side == 'SELL':
                        self.position -= filled_qty
                        self.balance += cost

                    avg_price = cost / filled_qty if filled_qty else 0
                    logging.info(f"Order FILLED: {trade_side} {filled_qty} {Config.TRADING_PAIR} at avg price {avg_price}. New Position: {self.position} {Config.BASE_CURRENCY}, New Balance: {self.balance} {Config.QUOTE_CURRENCY}")
                    print(f"PAPER TRADE: Order simulated - Filled {filled_qty} at avg price {avg_price}. New Position: {self.position} {Config.BASE_CURRENCY}, New Balance: {self.balance} {Config.QUOTE_CURRENCY}")

                else:
                    logging.warning(f"Order FILLED but no fills found in response: {order_response}")
                    print(f"Order FILLED but no fills found in response: {order_response}")

            else:
                logging.info(f"Order status: {order_response['status']} for order: {order_response['orderId']}")
                print(f"Order status: {order_response['status']} for order: {order_response['orderId']}")

        except Exception as e:
            logging.error(f"Error processing order fill: {e}. Order response: {order_response}")
            print(f"Error processing order fill: {e}. Order response: {order_response}")
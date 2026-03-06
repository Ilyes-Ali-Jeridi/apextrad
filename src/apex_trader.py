import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import asyncio
import websockets
import json
import logging

from decimal import Decimal
from binance.client import AsyncClient
from binance.streams import BinanceSocketManager
from collections import deque

from src.config import Config
from src.quantum_model import QuantumTemporalFusion
from src.ppo_agent import PPOTrader
from src.execution_engine import QuantumExecutionEngine
from src.feature_engine import QuantumFeatureEngine


# Configure logging
logging.basicConfig(filename='quantum_trader.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ApexQuantumTrader:
    """
    The main class for the Apex Quantum Trading system.

    This class orchestrates the entire trading process, from data ingestion and
    feature engineering to model prediction and trade execution. It integrates
    the different components of the system: the feature engine, the quantum
    model, the PPO agent, and the execution engine.
    """
    def __init__(self, api_key: str, api_secret: str):
        """
        Initializes the ApexQuantumTrader.

        Args:
            api_key (str): The Binance API key.
            api_secret (str): The Binance API secret.
        """
        self.model = QuantumTemporalFusion()
        self.engine = QuantumExecutionEngine(api_key, api_secret)
        self.feature_engine = QuantumFeatureEngine()
        self.ppo = PPOTrader(self.model)

        # Training state
        self.historical_data = deque(maxlen=10080)
        self.performance_metrics = {
            'daily_return': [],
            'sharpe': [],
            'max_drawdown': []
        }

        # Initialize with pre-trained weights (or not)
        # self._load_weights('quantum_weights.h5')

    async def run_strategy(self):
        """
        The main trading loop of the bot.

        This function connects to the Binance websockets for trade and depth
        data, processes the incoming data, generates features, makes predictions
        using the quantum model, and executes trades based on the model's output.
        It also handles the training of the model in an online fashion.
        """
        await self.engine.initialize()
        wm = self.engine.ws_manager
        trade_ws = wm.trade_socket(Config.TRADING_PAIR.lower())
        book_ticker_ws = wm.book_ticker_socket(Config.TRADING_PAIR.lower())
        ws_sockets = [trade_ws, book_ticker_ws]

        async with wm, trade_ws, book_ticker_ws:
            while True:
                try:
                    # Wait for and process trade and book ticker data
                    msg = await asyncio.wait_for(trade_ws.recv(), timeout=30)
                    trade_data = self._process_trade(msg)
                    if not trade_data:
                        continue

                    msg_book_ticker = await asyncio.wait_for(book_ticker_ws.recv(), timeout=30)
                    book_ticker_data = self._process_book_ticker(msg_book_ticker)
                    if not book_ticker_data:
                        continue

                    # Combine trade and book ticker data into a single tick
                    tick_data = {
                        'open': trade_data['open'],
                        'high': trade_data['high'],
                        'low': trade_data['low'],
                        'close': trade_data['close'],
                        'volume': trade_data['volume'],
                        'bid': book_ticker_data.get('bid', trade_data['close']),
                        'ask': book_ticker_data.get('ask', trade_data['close']),
                    }

                    # Generate features from the tick data
                    features = self.feature_engine.process_market_data(tick_data)
                    if features is None:
                        continue

                    # Make a prediction with the model using executor thread
                    pred = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.model(np.expand_dims(features, axis=0))
                    )

                    # Execute a trade if uncertainty is below the threshold
                    if pred['uncertainty'].numpy()[0][0] < Config.UNCERTAINTY_THRESHOLD:
                        action = self.ppo.sample_action(features)
                        await self._execute_strategy(action, pred)

                        # Store the experience in the PPO buffer
                        self.ppo.buffer['states'].append(features.numpy())
                        self.ppo.buffer['actions'].append(action.numpy())
                        self.ppo.buffer['rewards'].append(0.0) # Placeholder for reward
                        value = pred['value'].numpy()[0][0]
                        self.ppo.buffer['values'].append(value)
                        logprob = tfp.distributions.Normal(pred['price_params'][:, 0, 0], 1.0).log_prob(action).numpy()
                        self.ppo.buffer['logprobs'].append(logprob)

                    # Train the model if the buffer is full
                    if len(self.ppo.buffer['states']) >= Config.BATCH_SIZE:
                        print("Training model...")
                        logging.info("Training model...")
                        self._train_model()
                        print("Model trained.")
                        logging.info("Model trained.")
                        # Clear the buffer after training
                        self.ppo.buffer['states'].clear()
                        self.ppo.buffer['actions'].clear()
                        self.ppo.buffer['rewards'].clear()
                        self.ppo.buffer['values'].clear()
                        self.ppo.buffer['logprobs'].clear()

                except asyncio.TimeoutError:
                    # Handle websocket timeouts by reconnecting
                    logging.warning("Websocket timeout. Reconnecting...")
                    print("Websocket timeout. Reconnecting...")
                    
                    # Properly close websockets
                    try:
                        await asyncio.gather(
                            *[ws.close() for ws in ws_sockets],
                            return_exceptions=True
                        )
                    except Exception as close_error:
                        logging.warning(f"Error closing websockets: {close_error}")

                    logging.info(f"Waiting a few seconds before reconnecting websockets...")
                    await asyncio.sleep(5)

                    trade_ws = wm.trade_socket(Config.TRADING_PAIR.lower())
                    book_ticker_ws = wm.book_ticker_socket(Config.TRADING_PAIR.lower())
                    ws_sockets = [trade_ws, book_ticker_ws]
                    logging.info("Websockets reconnected.")
                    print("Websockets reconnected.")

                except Exception as e:
                    logging.error(f"Main loop error: {e}", exc_info=True)
                    print(f"Main loop error: {e}")
                    await asyncio.sleep(1)

    async def _execute_strategy(self, action: float, prediction: dict):
        """
        Executes the trading strategy based on the model's prediction and the
        PPO agent's action.

        Args:
            action (float): The action sampled from the PPO agent.
            prediction (dict): The output from the quantum model.
        """
        try:
            # Deconstruct the prediction
            mu, sigma, direction = tf.split(prediction['price_params'][0], 3, axis=-1)
            mu = mu.numpy()[0]
            sigma = sigma.numpy()[0]
            direction = direction.numpy()[0]

            # Calculate the position size with risk management
            position_size = self._calculate_position_size(action)

            # Execute a BUY or SELL order based on the predicted direction
            if direction > 0.6:
                logging.info(f"Strategy: BUY signal detected (direction={direction:.3f} > 0.6, uncertainty={prediction['uncertainty'].numpy()[0][0]:.3f}). Executing BUY order.")
                print(f"Strategy: BUY signal detected (direction={direction:.3f} > 0.6). Executing BUY order.")
                await self.engine.execute_order('BUY', position_size, mu - 1.5 * sigma)
            elif direction < 0.4:
                logging.info(f"Strategy: SELL signal detected (direction={direction:.3f} < 0.4, uncertainty={prediction['uncertainty'].numpy()[0][0]:.3f}). Executing SELL order.")
                print(f"Strategy: SELL signal detected (direction={direction:.3f} < 0.4). Executing SELL order.")
                await self.engine.execute_order('SELL', position_size, mu + 1.5 * sigma)
            else:
                logging.info(f"Strategy: No clear signal (direction={direction:.3f} between 0.4 and 0.6). Holding position.")
                print(f"Strategy: No clear signal (direction={direction:.3f}). Holding position.")

        except Exception as e:
            logging.error(f"Error in _execute_strategy: {e}")
            print(f"Error in _execute_strategy: {e}")

    def _calculate_position_size(self, action: float) -> float:
        """
        Calculates the position size based on the PPO agent's action and risk
        management rules.

        Args:
            action (float): The action from the PPO agent.

        Returns:
            float: The calculated position size.
        """
        # Apply risk management
        risk_adjusted_action = np.clip(action * Config.RISK_LEVERAGE, -Config.MAX_POSITION, Config.MAX_POSITION)
        balance_float = float(self.engine.balance)
        position_size = balance_float * abs(risk_adjusted_action) * Config.BASE_POSITION
        
        # Cap the position size at the risk limit
        max_risk = balance_float * Config.RISK_CAP
        position_size = min(position_size, max_risk)
        
        logging.info(f"Position size calculated: {position_size} (action={action:.3f}, balance={balance_float})")
        return position_size

    def _train_model(self):
        """
        Trains the PPO model using the buffered experiences.
        """
        try:
            states = np.array(self.ppo.buffer['states'])
            actions = np.array(self.ppo.buffer['actions'])
            rewards = np.array(self.ppo.buffer['rewards'])
            values = np.array(self.ppo.buffer['values'])
            logprobs = np.array(self.ppo.buffer['logprobs'])
            
            # Train the PPO agent
            self.ppo.train(states, actions, rewards, values, logprobs)
            
        except Exception as e:
            logging.error(f"Error in _train_model: {e}")
            print(f"Error in _train_model: {e}")

    def _process_trade(self, msg: str) -> dict | None:
        """
        Processes a trade message from the websocket.

        Args:
            msg (str): The raw websocket message.

        Returns:
            dict | None: The processed trade data, or None if processing fails.
        """
        try:
            event = json.loads(msg)
            if event['e'] != 'trade':
                return None

            return {
                'open': float(event['p']),
                'high': float(event['p']),
                'low': float(event['p']),
                'close': float(event['p']),
                'volume': float(event['q']),
                'bid': float(event['p']),
                'ask': float(event['p'])
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(f"Error processing trade message: {e}")
            return None

    def _process_book_ticker(self, msg: str) -> dict | None:
        """
        Processes a book ticker message from the websocket.

        Args:
            msg (str): The raw websocket message.

        Returns:
            dict | None: The processed book ticker data, or None if processing fails.
        """
        try:
            event = json.loads(msg)
            return {
                'bid': float(event.get('b', 0)),
                'ask': float(event.get('a', 0))
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(f"Error processing book ticker message: {e}")
            return None
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

                    # Make a prediction with the model
                    pred = self.model(np.expand_dims(features, axis=0))

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
                    for ws in ws_sockets:
                        await ws.close()

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
                logging.info(f"Strategy: Neutral signal (direction={direction:.3f}, uncertainty={prediction['uncertainty'].numpy()[0][0]:.3f}). No order executed.")
                print(f"Strategy: Neutral signal (direction={direction:.3f}). No order executed.")

        except Exception as e:
            logging.error(f"Error executing strategy: {e}", exc_info=True)
            print(f"Error executing strategy: {e}")

    def _calculate_position_size(self, action: float) -> float:
        """
        Calculates the position size for a trade, adjusted for risk.

        Args:
            action (float): The action from the PPO agent, which influences the
                            risk adjustment.

        Returns:
            float: The calculated position size.
        """
        base_size = Config.BASE_POSITION * self.engine.balance
        risk_adjusted = base_size * (1 + action * Config.RISK_LEVERAGE)
        position_size = min(risk_adjusted, Config.MAX_POSITION)
        logging.debug(f"Calculated position size: base_size={base_size}, risk_adjusted={risk_adjusted}, final_size={position_size}")
        return position_size

    def _train_model(self):
        """
        Trains the model using a hybrid approach that combines a price
        prediction loss with a policy loss from the PPO agent.
        """
        if not self.ppo.buffer['states']:
            logging.info("No data in buffer to train on. Skipping training.")
            return

        # Convert buffer deques to numpy arrays
        states = np.array(list(self.ppo.buffer['states']))
        actions = np.array(list(self.ppo.buffer['actions']))
        rewards = np.array(list(self.ppo.buffer['rewards']))

        # Ensure arrays have the correct dimensions
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, axis=0)
        if rewards.ndim == 1:
            rewards = np.expand_dims(rewards, axis=0)

        with tf.GradientTape() as tape:
            # Get model predictions for the states in the buffer
            pred = self.model(states)

            # Calculate the price prediction loss (MSE)
            price_loss = tf.keras.losses.MSE(
                pred['price_params'][:, 0, 0],
                np.array(list(self.feature_engine.close)[-len(states):])
            )

            # Calculate advantages and the PPO policy loss
            advantages = self.ppo.calculate_advantages(rewards, pred['value'][-1].numpy()[0])
            policy_loss = self.ppo.update_policy(states, actions, rewards)

            # Combine the losses
            total_loss = 0.7 * price_loss + 0.3 * policy_loss

        # Apply gradients to update the model's weights
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.ppo.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        logging.info(f"Model trained. Total Loss: {total_loss.numpy():.4f}, Price Loss: {price_loss.numpy():.4f}, Policy Loss: {policy_loss.numpy():.4f}")
        print(f"Model trained. Total Loss: {total_loss.numpy():.4f}, Price Loss: {price_loss.numpy():.4f}, Policy Loss: {policy_loss.numpy():.4f}")

    def _load_weights(self, path: str):
        """
        Loads pre-trained weights for the model.

        Args:
            path (str): The path to the weights file.
        """
        try:
            self.model.load_weights(path)
            logging.info(f"Loaded pretrained weights from {path}")
        except:
            logging.info("Initializing new model without pretrained weights.")

    def _process_trade(self, msg: str) -> dict | None:
        """
        Processes a trade message from the Binance websocket.

        Args:
            msg (str): The raw websocket message.

        Returns:
            dict | None: A dictionary containing the trade data, or None if the
                         message is not a trade event.
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
        except Exception as e:
            logging.error(f"Error processing trade message: {e}", exc_info=True)
            return None

    def _process_book_ticker(self, msg: str) -> dict | None:
        """
        Processes a book ticker message from the Binance websocket.

        Args:
            msg (str): The raw websocket message.

        Returns:
            dict | None: A dictionary containing the best bid and ask, or None
                         if the message cannot be parsed.
        """
        try:
            event = json.loads(msg)
            return {
                'bid': float(event['b']),
                'ask': float(event['a'])
            }
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error processing book ticker message: {e}", exc_info=True)
            return None


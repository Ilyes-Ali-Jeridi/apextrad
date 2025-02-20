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

from .config import Config # Import Config from config.py
from .quantum_model import QuantumTemporalFusion # Import QuantumTemporalFusion from quantum_model.py
from .ppo_agent import PPOTrader # Import PPOTrader from ppo_agent.py
from .execution_engine import QuantumExecutionEngine # Import QuantumExecutionEngine from execution_engine.py
from .feature_engine import QuantumFeatureEngine # Import QuantumFeatureEngine from feature_engine.py


# Configure logging
logging.basicConfig(filename='quantum_trader.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ApexQuantumTrader:
    """End-to-End Quantum Trading System"""
    def __init__(self, api_key: str, api_secret: str):
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
        """Main trading loop - fetches data, predicts, and executes trades."""
        await self.engine.initialize()
        wm = self.engine.ws_manager
        trade_ws = wm.trade_socket(Config.TRADING_PAIR.lower())
        depth_ws = wm.depth_socket(Config.TRADING_PAIR.lower())
        ws_sockets = [trade_ws, depth_ws]

        async with wm, trade_ws, depth_ws:
            while True:
                try:
                    msg = await asyncio.wait_for(trade_ws.recv(), timeout=30)
                    trade_data = self._process_trade(msg)
                    if not trade_data:
                        continue

                    msg_depth = await asyncio.wait_for(depth_ws.recv(), timeout=30)
                    depth_data = self._process_depth(msg_depth)
                    if not depth_data:
                        continue

                    tick_data = {
                        'open': trade_data['open'],
                        'high': trade_data['high'],
                        'low': trade_data['low'],
                        'close': trade_data['close'],
                        'volume': trade_data['volume'],
                        'bid': depth_data.get('bid', trade_data['close']),
                        'ask': depth_data.get('ask', trade_data['close']),
                    }

                    features = self.feature_engine.process_market_data(tick_data)
                    if features is None:
                        continue

                    pred = self.model(np.expand_dims(features, axis=0))
                    if pred['uncertainty'].numpy()[0][0] < Config.UNCERTAINTY_THRESHOLD:
                        action = self.ppo.sample_action(features)
                        await self._execute_strategy(action, pred)
                        self.ppo.buffer['states'].append(features.numpy())
                        self.ppo.buffer['actions'].append(action.numpy())
                        self.ppo.buffer['rewards'].append(0.0)
                        value = pred['value'].numpy()[0][0]
                        self.ppo.buffer['values'].append(value)
                        logprob = tfp.distributions.Normal(pred['price_params'][:, 0, 0], 1.0).log_prob(action).numpy()
                        self.ppo.buffer['logprobs'].append(logprob)

                    if len(self.ppo.buffer['states']) >= Config.BATCH_SIZE:
                        print("Training model...")
                        logging.info("Training model...")
                        self._train_model()
                        print("Model trained.")
                        logging.info("Model trained.")
                        self.ppo.buffer['states'].clear()
                        self.ppo.buffer['actions'].clear()
                        self.ppo.buffer['rewards'].clear()
                        self.ppo.buffer['values'].clear()
                        self.ppo.buffer['logprobs'].clear()

                except asyncio.TimeoutError:
                    logging.warning("Websocket timeout. Reconnecting...")
                    print("Websocket timeout. Reconnecting...")
                    for ws in ws_sockets:
                        await ws.close()
                    wait_time = 2**ws_sockets.index(trade_ws)
                    logging.info(f"Waiting {wait_time} seconds before reconnecting websockets...")
                    await asyncio.sleep(wait_time)

                    trade_ws = wm.trade_socket(Config.TRADING_PAIR.lower())
                    depth_ws = wm.depth_socket(Config.TRADING_PAIR.lower())
                    ws_sockets = [trade_ws, depth_ws]
                    logging.info("Websockets reconnected.")
                    print("Websockets reconnected.")

                except Exception as e:
                    logging.error(f"Main loop error: {e}", exc_info=True)
                    print(f"Main loop error: {e}")
                    await asyncio.sleep(1)

    async def _execute_strategy(self, action: float, prediction: dict):
        """Executes trading strategy based on action and prediction."""
        try:
            mu, sigma, direction = tf.split(prediction['price_params'][0], 3, axis=-1)
            mu = mu.numpy()[0]
            sigma = sigma.numpy()[0]
            direction = direction.numpy()[0]
            position_size = self._calculate_position_size(action)

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

    def _calculate_position_size(self, action: float):
        """Risk-managed size calculation."""
        base_size = Config.BASE_POSITION * self.engine.balance
        risk_adjusted = base_size * (1 + action * Config.RISK_LEVERAGE)
        position_size = min(risk_adjusted, Config.MAX_POSITION)
        logging.debug(f"Calculated position size: base_size={base_size}, risk_adjusted={risk_adjusted}, final_size={position_size}")
        return position_size

    def _train_model(self):
        """Hybrid training procedure."""
        if not self.ppo.buffer['states']:
            logging.info("No data in buffer to train on. Skipping training.")
            return

        states = np.array(list(self.ppo.buffer['states']))
        actions = np.array(list(self.ppo.buffer['actions']))
        rewards = np.array(list(self.ppo.buffer['rewards']))

        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, axis=0)
        if rewards.ndim == 1:
            rewards = np.expand_dims(rewards, axis=0)

        with tf.GradientTape() as tape:
            pred = self.model(states)
            price_loss = tf.keras.losses.MSE(
                pred['price_params'][:, 0, 0],
                np.array(list(self.feature_engine.close)[-len(states):])
            )

            advantages = self.ppo.calculate_advantages(rewards, pred['value'][-1].numpy()[0])
            policy_loss = self.ppo.update_policy(states, actions, rewards)

            total_loss = 0.7 * price_loss + 0.3 * policy_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.ppo.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        logging.info(f"Model trained. Total Loss: {total_loss.numpy():.4f}, Price Loss: {price_loss.numpy():.4f}, Policy Loss: {policy_loss.numpy():.4f}")
        print(f"Model trained. Total Loss: {total_loss.numpy():.4f}, Price Loss: {price_loss.numpy():.4f}, Policy Loss: {policy_loss.numpy():.4f}")

    def _load_weights(self, path: str):
        """Loads pre-trained model weights."""
        try:
            self.model.load_weights(path)
            logging.info(f"Loaded pretrained weights from {path}")
        except:
            logging.info("Initializing new model without pretrained weights.")

    def _process_trade(self, msg):
        """Processes trade websocket message."""
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

    def _process_depth(self, msg):
        """Processes depth websocket message for bid/ask."""
        try:
            event = json.loads(msg)
            if event['e'] != 'depthUpdate':
                return None

            bids = event['b']
            asks = event['a']

            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None

            return {
                'bid': best_bid,
                'ask': best_ask
            }
        except Exception as e:
            logging.error(f"Error processing depth message: {e}", exc_info=True)
            return {'bid': None, 'ask': None}


if __name__ == "__main__":
    api_key = os.getenv('BINANCE_KEY')
    api_secret = os.getenv('BINANCE_SECRET')

    if not api_key or not api_secret:
        print("Error: Binance API keys not found in environment variables BINANCE_KEY and BINANCE_SECRET.")
        print("Please set these environment variables before running the script.")
        exit()

    trader = ApexQuantumTrader(api_key, api_secret)
    loop = asyncio.new_event_loop()

    try:
        print("Starting Quantum Trading System...")
        logging.info("Starting Quantum Trading System...")
        loop.run_until_complete(trader.run_strategy())
    except KeyboardInterrupt:
        print("\nGraceful shutdown initiated")
        logging.info("Graceful shutdown initiated")
    except Exception as e:
        print(f"Critical error in main loop: {e}")
        logging.critical(f"Critical error in main loop: {e}", exc_info=True)
    finally:
        loop.close()
        print("Quantum session terminated")
        logging.info("Quantum session terminated")
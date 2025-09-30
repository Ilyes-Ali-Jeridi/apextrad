import numpy as np
import pandas as pd
import tensorflow as tf
import talib
from collections import deque
import logging

class QuantumFeatureEngine:
    """
    Handles the real-time processing of market data to generate features for
    the trading model. It uses rolling windows to maintain a recent history of
    market data and calculates various technical indicators.
    """
    def __init__(self):
        """
        Initializes the QuantumFeatureEngine.

        This sets up the rolling windows for market data and normalizers for
        scaling the features.
        """
        self.window_size = 390
        self._init_rolling_windows()

        # Initialize normalizers for price and volume
        self.scalers = {
            'price': tf.keras.layers.Normalization(),
            'volume': tf.keras.layers.Normalization()
        }
        # Adapt the normalizers with some dummy data to avoid errors on startup
        dummy_price_data = np.random.rand(self.window_size, 1)
        dummy_volume_data = np.random.rand(self.window_size, 1)
        self.scalers['price'].adapt(dummy_price_data)
        self.scalers['volume'].adapt(dummy_volume_data)

    def _init_rolling_windows(self):
        """
        Initializes the rolling windows (deques) to store historical market
        data for feature calculation.
        """
        self.open = deque(maxlen=self.window_size)
        self.high = deque(maxlen=self.window_size)
        self.low = deque(maxlen=self.window_size)
        self.close = deque(maxlen=self.window_size)
        self.volume = deque(maxlen=self.window_size)
        self.vpin = deque(maxlen=50)

    def process_market_data(self, tick: dict) -> tf.Tensor | None:
        """
        Processes a single tick of market data to generate a feature vector.

        Args:
            tick (dict): A dictionary containing the latest market data,
                         including open, high, low, close, volume, bid, and ask.

        Returns:
            tf.Tensor | None: A TensorFlow tensor representing the feature
                              vector, or None if an error occurs.
        """
        try:
            # Append the latest data to the rolling windows
            self.open.append(float(tick['open']))
            self.high.append(float(tick['high']))
            self.low.append(float(tick['low']))
            self.close.append(float(tick['close']))
            self.volume.append(float(tick['volume']))
        except KeyError as e:
            logging.warning(f"KeyError in market data: {e}. Skipping tick.")
            return None

        # Calculate various features
        features = {}
        try:
            features = {
                'vpin': self._calculate_vpin(),
                'obv': talib.OBV(np.array(list(self.close)), np.array(list(self.volume)))[-1] if self.close and self.volume else 0.0,
                'spread': (float(tick['ask']) - float(tick['bid'])),
                'mid_price': (float(tick['ask']) + float(tick['bid'])) / 2,
                'volume_profile': np.log(self.volume[-1] / np.mean(list(self.volume))) if self.volume and np.mean(list(self.volume)) > 0 and self.volume[-1] > 0 else 0.0
            }
        except Exception as e:
            logging.error(f"Feature calculation error: {e}. Returning None features.")
            return None

        try:
            # Scale and concatenate the features into a single tensor
            price_feature = np.array([[features['mid_price']]])
            scaled_price = self.scalers['price'](price_feature) if features['mid_price'] is not None else 0.0

            vpin_feature = features['vpin'] if features['vpin'] is not None else 0.0
            volume_profile_feature = features['volume_profile'] if features['volume_profile'] is not None else 0.0

            processed_features = tf.concat([
                tf.reshape(scaled_price, [-1]),
                tf.expand_dims(vpin_feature, axis=0) if isinstance(vpin_feature, (int, float)) else tf.zeros((1,)),
                tf.expand_dims(volume_profile_feature, axis=0) if isinstance(volume_profile_feature, (int, float)) else tf.zeros((1,))
            ], axis=-1)
            logging.debug(f"Processed features: {processed_features.numpy()}")
            return processed_features

        except Exception as e:
            logging.error(f"Error during feature scaling/concatenation: {e}. Returning None features.")
            return None

    def _calculate_vpin(self) -> float:
        """
        Calculates the Volume-Synchronized Probability of Informed Trading (VPIN).

        VPIN is a measure of order flow imbalance, which can indicate the
        presence of informed traders.

        Returns:
            float: The calculated VPIN value.
        """
        if len(self.close) < 50:
            return 0.0

        buys = sells = 0
        for i in range(max(0, len(self.close) - 50), len(self.close)):
            if self.close[i] > self.open[i]:
                buys += self.volume[i]
            else:
                sells += self.volume[i]

        total_volume = buys + sells
        vpin_value = (buys - sells) / (total_volume + 1e-8) if total_volume > 0 else 0.0
        logging.debug(f"VPIN calculated: {vpin_value}")
        return vpin_value
    async def process_market_data(self, tick: dict) -> tf.Tensor | None:
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

        # Calculate features asynchronously where needed
        features = {}
        try:
            # Calculate OBV in thread pool to avoid blocking
            obv = await asyncio.get_event_loop().run_in_executor(
                ta_executor, _safe_obv, 
                np.array(list(self.close)), 
                np.array(list(self.volume))
            )
            features = {
                'vpin': self._calculate_vpin(),
                'obv': float(obv[-1]) if obv.size > 0 else 0.0,
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
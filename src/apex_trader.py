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
            direction = float(direction.numpy()[0])
            
            # Clamp direction to [0, 1] range and validate
            if direction < 0.0 or direction > 1.0:
                logging.warning(f"Direction {direction} outside expected range [0, 1], clamping.")
                direction = np.clip(direction, 0.0, 1.0)
            
            # Calculate the position size with risk management
            position_size = self._calculate_position_size(action)

            # Execute a BUY or SELL order based on the predicted direction
            if direction > 0.6:
                logging.info(f"Strategy: BUY signal detected (direction={direction:.3f} > 0.6, uncertainty={prediction['uncertainty'].numpy()[0][0]:.3f}). Executing BUY order.")
                print(f"Strategy: BUY signal detected (direction={direction:.3f} > 0.6). Executing BUY order.")
                await self.engine.execute_order('BUY', position_size, mu - 1.5 * sigma)
            elif direction < 0.4:
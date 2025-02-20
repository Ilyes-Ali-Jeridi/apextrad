import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque

class PPOTrader:
    """Proximal Policy Optimization Agent with Experience Replay"""
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.optimizer = tf.keras.optimizers.Adam(3e-4)
        self.buffer = {
            'states': deque(maxlen=32768),
            'actions': deque(maxlen=32768),
            'rewards': deque(maxlen=32768),
            'values': deque(maxlen=32768),
            'logprobs': deque(maxlen=32768)
        }
        self.lam = 0.95

    @tf.function
    def sample_action(self, state):
        """Samples an action from the model."""
        output = self.model(state[None, :], training=True)
        mu, var, _ = tf.split(output['price_params'], 3, axis=-1)
        return tf.squeeze(mu + tf.sqrt(var) * tf.random.normal(mu.shape))

    def calculate_advantages(self, rewards, last_value):
        """Calculates advantages using GAE."""
        advantages = np.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (last_value if t == len(rewards) - 1 else self.buffer['values'][t + 1]) - self.buffer['values'][t]
            advantages[t] = delta + self.gamma * self.lam * last_adv
            last_adv = advantages[t]
        return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    @tf.function
    def update_policy(self, states, actions, rewards):
        """Updates the policy using PPO algorithm."""
        with tf.GradientTape() as tape:
            new_output = self.model(states, training=True)
            new_logits = new_output['price_params'][:, 0]
            new_value = new_output['value'][:, 0]

            ratio = tf.exp(tfp.distributions.Normal(new_logits, 1.0).log_prob(actions) -
                           tfp.distributions.Normal(self.buffer['logprobs'], 1.0).log_prob(actions))

            clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            advantages = self.calculate_advantages(rewards, new_value[-1])
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))

            value_loss = 0.5 * tf.reduce_mean((new_value - rewards) ** 2)
            entropy_loss = -0.01 * tf.reduce_mean(tfp.distributions.Normal(new_logits, 1.0).entropy())

            total_loss = policy_loss + value_loss + entropy_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque

class PPOTrader:
    """
    An agent that uses the Proximal Policy Optimization (PPO) algorithm to
    learn a trading policy. It includes an experience replay buffer for storing
    and sampling experiences.
    """
    def __init__(self, model: tf.keras.Model):
        """
        Initializes the PPOTrader agent.

        Args:
            model (tf.keras.Model): The policy and value network model.
        """
        self.model = model
        self.gamma = 0.99  # Discount factor for future rewards
        self.clip_ratio = 0.2  # Clipping parameter for the PPO objective function
        self.optimizer = tf.keras.optimizers.Adam(3e-4)
        self.buffer = {
            'states': deque(maxlen=32768),
            'actions': deque(maxlen=32768),
            'rewards': deque(maxlen=32768),
            'values': deque(maxlen=32768),
            'logprobs': deque(maxlen=32768)
        }
        self.lam = 0.95  # Lambda for Generalized Advantage Estimation (GAE)

    @tf.function
    def sample_action(self, state: np.ndarray) -> tf.Tensor:
        """
        Samples an action from the policy network for a given state.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            tf.Tensor: The action sampled from the policy.
        """
        output = self.model(state[None, :], training=True)
        mu, var, _ = tf.split(output['price_params'], 3, axis=-1)
        return tf.squeeze(mu + tf.sqrt(tf.maximum(var, 1e-8)) * tf.random.normal(mu.shape))

    def calculate_advantages(self, rewards: np.ndarray, last_value: float) -> np.ndarray:
        """
        Calculates the advantages for a batch of experiences using Generalized
        Advantage Estimation (GAE).

        Args:
            rewards (np.ndarray): The rewards received during the batch.
            last_value (float): The value of the last state in the batch, as
                                estimated by the value network.

        Returns:
            np.ndarray: The calculated advantages.
        """
        advantages = np.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (last_value if t == len(rewards) - 1 else self.buffer['values'][t + 1]) - self.buffer['values'][t]
            advantages[t] = delta + self.gamma * self.lam * last_adv
            last_adv = advantages[t]
        # Normalize the advantages
        return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    @tf.function
    def update_policy(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> tf.Tensor:
        """
        Updates the policy and value networks using the PPO algorithm.

        Args:
            states (np.ndarray): A batch of states.
            actions (np.ndarray): A batch of actions.
            rewards (np.ndarray): A batch of rewards.

        Returns:
            tf.Tensor: The total loss for the update step.
        """
        with tf.GradientTape() as tape:
            new_output = self.model(states, training=True)
            new_logits = new_output['price_params'][:, 0]
            new_value = new_output['value'][:, 0]

            # Calculate the ratio of the new and old policies
            ratio = tf.exp(tfp.distributions.Normal(new_logits, 1.0).log_prob(actions) -
                           tf.constant(np.array(list(self.buffer['logprobs'])), dtype=tf.float32))

            # Clip the ratio
            clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            advantages = self.calculate_advantages(rewards, new_value[-1])

            # Calculate the PPO policy loss
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))

            # Calculate the value loss
            value_loss = 0.5 * tf.reduce_mean((new_value - rewards) ** 2)

            # Calculate the entropy loss to encourage exploration
            entropy_loss = -0.01 * tf.reduce_mean(tfp.distributions.Normal(new_logits, 1.0).entropy())

            # Combine the losses
            total_loss = policy_loss + value_loss + entropy_loss

        # Apply the gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss
import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow_probability as tfp
import cirq
import numpy as np

class QuantumTemporalFusion(tf.keras.Model):
    """
    A hybrid quantum-classical model for time-series forecasting.

    This model integrates a Temporal Convolutional Network (TCN) to capture
    temporal dependencies, a Parameterized Quantum Circuit (PQC) to introduce
    quantum computational features, and a temporal attention mechanism to
    focus on relevant time steps. It also includes a Bayesian dense layer for
    uncertainty quantification.
    """
    def __init__(self, num_features: int = 3):
        """
        Initializes the QuantumTemporalFusion model.

        Args:
            num_features (int): The number of input features.
        """
        super().__init__()

        # Temporal Convolutional Network (TCN) for feature extraction from time series
        self.tcn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, num_features)),
            tf.keras.layers.Conv1D(128, 5, dilation_rate=1, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(256, 5, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ], name='tcn_layers')

        # Quantum components
        self.qubits = cirq.GridQubit.rect(1, 4)
        self.quantum_layer = tfq.layers.PQC(
            self._build_quantum_circuit(),
            [cirq.Z(q) for q in self.qubits],  # Define observables
            name='pqc_layer'
        )

        # Temporal Attention layer
        self.temporal_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, name='temporal_attention')

        # Bayesian Dense layer for uncertainty estimation
        self.bayesian_dense = tfp.layers.DenseVariational(
            units=64,
            activation='relu',
            make_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            name='bayesian_dense'
        )

        # Output heads for different tasks
        self.price_head = tf.keras.layers.Dense(units=3, name='price_head')  # e.g., mu, sigma, direction
        self.regime_head = tf.keras.layers.Dense(units=4, activation='softmax', name='regime_head')  # e.g., market regimes
        self.value_head = tf.keras.layers.Dense(units=1, name='value_head')  # For PPO value function
        self.uncertainty_head = tf.keras.layers.Dense(units=2, activation='sigmoid', name='uncertainty_head')

    def _build_quantum_circuit(self) -> cirq.Circuit:
        """
        Builds the Parameterized Quantum Circuit (PQC).

        Returns:
            cirq.Circuit: The constructed quantum circuit.
        """
        circuit = cirq.Circuit()
        # Add a layer of Hadamard gates
        circuit.append(cirq.H.on_each(self.qubits))
        # Add entangling layers
        circuit.append(cirq.CNOT(self.qubits[0], self.qubits[1]))
        circuit.append(cirq.CNOT(self.qubits[2], self.qubits[3]))
        circuit.append(cirq.CZ(self.qubits[1], self.qubits[2]))
        return circuit

    def call(self, inputs: tf.Tensor, training: bool = False) -> dict:
        """
        The forward pass of the model.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, time_steps, num_features).
            training (bool): Whether the model is in training mode.

        Returns:
            dict: A dictionary of output tensors for different tasks.
        """
        # Ensure input has the right shape
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)

        # Process inputs through the TCN
        tcn_out = self.tcn(inputs)

        # Process the TCN output through the quantum layer
        # We need to flatten the output of the TCN to be compatible with the PQC layer
        batch_size = tf.shape(tcn_out)[0]
        tcn_out_reshaped = tf.reshape(tcn_out, [batch_size, -1])

        # Trim or pad to fit the number of parameters for the quantum circuit
        num_params = np.prod(self.quantum_layer.symbols.shape)
        tcn_out_padded = tf.pad(tcn_out_reshaped, [[0, 0], [0, max(0, num_params - tcn_out_reshaped.shape[1])]])
        quantum_in = tcn_out_padded[:, :num_params]

        quantum_out = self.quantum_layer(quantum_in)

        # Process TCN output through the attention layer
        attn_out = self.temporal_attention(tcn_out, tcn_out)
        attn_out = tf.reduce_mean(attn_out, axis=1) # Aggregate attention outputs

        # Fuse the classical and quantum outputs
        fused = tf.concat([attn_out, quantum_out], axis=-1)
        fused = self.bayesian_dense(fused)

        # Return the outputs from the different heads
        return {
            'price_params': self.price_head(fused),
            'regime': self.regime_head(fused),
            'value': self.value_head(fused),
            'uncertainty': self.uncertainty_head(fused)
        }
import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow_probability as tfp
import cirq
import numpy as np

class QuantumTemporalFusion(tf.keras.Model):
    """
    Hybrid Quantum-Classical Temporal Model for price prediction.
    Combines Temporal Convolutional Networks (TCNs), Quantum Neural Networks (PQC),
    and Temporal Attention mechanisms to capture complex temporal dependencies in market data.
    """
    def __init__(self, num_features: int = 21):  # num_features adjusted to 21 as news feature is removed
        super().__init__()

        # Temporal Convolutional Network (TCN) layers to process sequential market data
        self.tcn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 5, dilation_rate=1, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(256, 5, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ], name='tcn_layers')

        # Define 4 qubits arranged in a 1x4 grid for the quantum layer
        self.qubits = cirq.GridQubit.rect(1, 4)
        # Parameterized Quantum Circuit (PQC) layer using TensorFlow Quantum
        self.quantum_layer = tfq.layers.PQC(
            self._build_quantum_circuit(),
            cirq.Z(self.qubits[0]) + cirq.X(self.qubits[1]),
            name='pqc_layer'
        )

        # Temporal Attention layer to weigh different time steps based on their relevance
        self.temporal_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, name='temporal_attention')
        # Bayesian Dense layer for uncertainty estimation, using TensorFlow Probability
        self.bayesian_dense = tfp.layers.DenseVariational(units=64, activation='relu', make_posterior_fn=tfp.layers.default_mean_field_normal_fn(), name='bayesian_dense')
        # Uncertainty head to predict uncertainty in price movement prediction
        self.uncertainty_head = tf.keras.layers.Dense(units=2, activation='sigmoid', name='uncertainty_head')

        # Output heads for different prediction tasks
        self.price_head = tf.keras.layers.Dense(units=3, name='price_head')
        self.regime_head = tf.keras.layers.Dense(units=4, activation='softmax', name='regime_head')
        self.value_head = tf.keras.layers.Dense(units=1, name='value_head')

    def _build_quantum_circuit(self):
        """Builds a simple quantum circuit using Cirq."""
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(self.qubits))
        circuit.append(cirq.CNOT(self.qubits[0], self.qubits[2]))
        circuit.append(cirq.CZ(self.qubits[1], self.qubits[3]))
        return circuit

    def call(self, inputs, training=False):
        """Forward pass of the QuantumTemporalFusion model."""
        tcn_out = self.tcn(inputs)
        quantum_out = self.quantum_layer(tcn_out)
        attn_out = self.temporal_attention(tcn_out, tcn_out)

        fused = quantum_out * 0.6 + attn_out * 0.4
        fused = self.bayesian_dense(fused)

        return {
            'price_params': self.price_head(fused),
            'regime': self.regime_head(fused),
            'value': self.value_head(fused),
            'uncertainty': self.uncertainty_head(fused)
        }
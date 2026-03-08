    def _build_quantum_execution_circuit(self):
        """
        Builds the parameterized quantum circuit for price adjustments.

        Returns:
            cirq.Circuit: The quantum circuit.
        """
        q0 = cirq.GridQubit(0, 0)
        spread_param = cirq.HashableParamExpr(cirq.Symbol('spread_param'))
        
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.rx(spread_param)(q0),
            cirq.measure(q0, key='m')
        )
        return circuit

    def _calculate_quantum_adjustment(self) -> float:
        """
        Calculates a price adjustment using a parameterized quantum circuit.
        The adjustment is based on the current bid-ask spread.

        Returns:
            float: The calculated price adjustment.
        """
        try:
            spread = (self.best_ask - self.best_bid) if hasattr(self, 'best_bid') and hasattr(self, 'best_ask') and self.best_bid and 
self.best_ask else Decimal('0.0001')
            scaled_spread = float(spread) / 100.0  # Normalize spread

            # Create param resolver with matching symbol name
            resolver = cirq.ParamResolver({'spread_param': scaled_spread})
            
            # Build expectation layer with matching symbols
            expectation_layer = tfq.layers.Expectation()
            expectation = expectation_layer(
                [tfq.convert_to_tensor([self.quantum_circuit])],
                symbol_names=['spread_param'],
                symbol_values=[[scaled_spread]]
            )
            
            exp_val = float(expectation.numpy()[0][0])
            adjustment = 0.5 * (1.0 - exp_val)  # Map [-1,1] to [0,1] adjustment factor
            
            logging.debug(f"Quantum adjustment: {adjustment} based on spread: {spread}")
            return adjustment
        except Exception as e:
            logging.error(f"Error in quantum adjustment calculation: {e}")
            return 0.0
# Apex Quantum Trader

Apex Quantum Trader is a sophisticated, end-to-end algorithmic trading bot that leverages a hybrid quantum-classical machine learning model to predict cryptocurrency price movements and execute trades on the Binance exchange. The system is designed for modularity and extensibility, allowing for easy customization of its components.

## Features

- **Hybrid Quantum-Classical Model**: Utilizes a `QuantumTemporalFusion` model that combines Temporal Convolutional Networks (TCNs), a Parameterized Quantum Circuit (PQC), and a temporal attention mechanism to capture complex market dynamics.
- **Reinforcement Learning**: Employs a Proximal Policy Optimization (PPO) agent to learn an optimal trading policy through online training.
- **Real-Time Data Processing**: Connects to Binance websockets for real-time trade and order book data.
- **Advanced Feature Engineering**: The `QuantumFeatureEngine` calculates a variety of technical indicators, including VPIN (Volume-Synchronized Probability of Informed Trading) and OBV (On-Balance Volume).
- **Risk Management**: The `QuantumExecutionEngine` includes robust risk management features, such as position sizing based on a configurable risk cap.
- **Paper and Live Trading**: The bot can be run in either paper trading mode for simulation or live trading mode with real funds.

## Project Structure

```
.
├── main.py                 # The main entry point to run the trading bot
├── requirements.txt        # A list of all the Python packages required for the project
├── README.md               # This file
└── src/
    ├── __init__.py         # Makes the 'src' directory a Python package
    ├── apex_trader.py      # The main class that orchestrates the trading system
    ├── config.py           # All configuration parameters for the bot
    ├── execution_engine.py # Handles order execution and risk management
    ├── feature_engine.py   # Processes market data and generates features
    ├── ppo_agent.py        # The PPO agent for reinforcement learning
    └── quantum_model.py    # The hybrid quantum-classical model for price prediction
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd apex-quantum-trader
    ```

2.  **Install the required dependencies:**
    It is recommended to use a virtual environment to avoid conflicts with other projects.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    **Note:** Installing `TA-Lib` can be tricky. Please refer to the official [TA-Lib installation guide](https://github.com/mrjbq7/ta-lib#installation) for detailed instructions for your operating system.

## Configuration

1.  **Set up Binance API keys:**
    The bot requires Binance API keys to access market data and execute trades. You need to set them as environment variables:
    ```bash
    export BINANCE_KEY="your_api_key"
    export BINANCE_SECRET="your_api_secret"
    ```

2.  **Configure the trading bot:**
    All configuration parameters are located in `src/config.py`. You can modify this file to change the trading pair, initial balance, risk management settings, and other parameters.

    - `PAPER_TRADING`: Set to `True` to run in paper trading mode (no real funds will be used). Set to `False` for live trading.
    - `TRADING_PAIR`: The trading pair you want the bot to trade (e.g., "BTCUSDT", "ETHUSDT").
    - `INITIAL_BALANCE`: The initial balance for paper trading.
    - `RISK_CAP`: The maximum percentage of your balance to risk on a single trade.

## Usage

To run the trading bot, execute the `main.py` script from the root directory of the project:

```bash
python main.py
```

The bot will start, connect to the Binance websockets, and begin processing market data. You will see log messages in the console and in the `quantum_trader.log` file.

## Disclaimer

Trading cryptocurrencies involves significant risk. This project is for educational purposes only and should not be used for live trading without a thorough understanding of the code and the risks involved. The authors are not responsible for any financial losses you may incur. Always do your own research and trade responsibly.
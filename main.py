import os
import asyncio
import logging
from src.apex_trader import ApexQuantumTrader

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
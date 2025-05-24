# RiskTrader RL

An AI-powered stock trading simulator using Q-Learning reinforcement learning agent.

## Features
- Historical stock data using `yfinance`
- Technical indicators like MA10, MA50, RSI
- Gym-like custom trading environment
- Q-Learning agent
- Visualization of performance

## Run the Project
```bash
pip install -r requirements.txt
python train.py
```

## Future Ideas
- Replace Q-Learning with DQN using PyTorch
- Add transaction fees, slippage
- Test multiple tickers (TSLA, GOOGL, etc.)
- Add paper trading integration
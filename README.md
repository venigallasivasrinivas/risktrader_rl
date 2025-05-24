# RiskTrader RL

This project is a **smart stock trading program** that learns how to buy and sell stocks to make money — all by itself!

---

## What is this project about?

This project uses **Reinforcement Learning (RL)**, which is a way computers learn from experience by trying actions and seeing what happens, just like how we learn by trial and error.

---

## Important concepts explained simply

### 1. Reinforcement Learning (RL)

- Imagine teaching a dog new tricks by giving it treats when it does something right.
- Here, the **computer (agent)** learns how to trade stocks by trying actions and getting rewards (like treats) when it makes money.
- The goal is to learn the best actions to make the most money over time.

### 2. Deep Q-Network (DQN)

- The agent needs to decide if it should **buy**, **hold**, or **sell** a stock at any time.
- DQN is like the agent’s brain — it’s a type of **neural network** (a computer model inspired by the brain) that learns to pick the best action based on what it sees.
- It looks at the current market and decides what will likely earn the most money.

### 3. Trading Environment

- This is the **world** where the agent trades stocks.
- It gives the agent information about the stock price and other details at each step.
- When the agent decides an action, the environment updates the portfolio (money and stocks owned) and tells the agent how well it did (reward).

### 4. Technical Indicators

- These are **simple math tools** that help the agent understand the stock market better.
- We use:
  - **MA10 and MA50 (Moving Averages):** They show the average stock price over the last 10 and 50 days to spot trends.
  - **RSI (Relative Strength Index):** Shows if a stock is overbought or oversold, hinting if the price might go up or down.
  - **Volume:** How many stocks were traded, showing how active the market is.

### 5. State

- This is what the agent **sees** at each moment.
- It's like a snapshot of the market, including current price and the technical indicators.

### 6. Actions

- The agent can choose from three:
  - **Buy:** Purchase stocks if it thinks price will go up.
  - **Hold:** Do nothing, wait and watch.
  - **Sell:** Sell owned stocks to take profit or stop loss.

### 7. Reward

- When the agent sells, it gets a **reward** equal to the profit or loss it made.
- The agent tries to make decisions that increase its total reward — meaning more money!

---

## How the program works

1. It gets **historical stock data** from Yahoo Finance.
2. Calculates technical indicators to help the agent understand market trends.
3. Trains the DQN agent on past data so it can learn good trading strategies.
4. Tests the agent on new, unseen data to see how well it performs.
5. Shows charts and stats to visualize the agent’s trading success.

---

## Why is this useful?

- It helps us understand how computers can learn to trade automatically.
- This can assist traders or help build smart financial tools.
- It’s a great way to learn about AI, finance, and programming all together!

---

## Screenshot

![AAPL Trading Results](./screenshots/aapl_trading_results.png)

*This shows how well the agent traded Apple stock after learning.*

---

## How to use

- Run the program.
- Enter a stock ticker (like `AAPL`).
- Watch the agent learn and trade the stock.
- Check the results and graphs.

---

## Project files

- `train.py` – Main file to train and test the agent.
- `trading_env.py` – The stock trading environment.
- `dqn_agent.py` – The AI agent that learns how to trade.
- `evaluate.py` – Tools to show charts and trading results.

---

Made by Venigalla Siva Srinivas. Feel free to ask questions or suggest improvements!
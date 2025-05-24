import yfinance as yf
import pandas as pd
from agent.dqn_agent import DQNAgent
from env.trading_env import TradingEnv
from evaluate import plot_trades, plot_rewards
import numpy as np
import os

def fetch_data(ticker="AAPL", start="2022-01-01", end="2023-12-31"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    symbol = input("Enter the stock ticker (e.g., AAPL): ").upper()
    data = fetch_data(ticker=symbol)

    if data.empty:
        print(f"No data fetched for ticker {symbol} in the given date range.")
        exit()

    train_data = data[data.index.year == 2022]
    test_data = data[data.index.year == 2023]

    if train_data.empty or test_data.empty:
        print("Training or testing data is empty. Please check your date ranges and ticker.")
        exit()

    actions = ["buy", "hold", "sell"]
    state_dim = 6
    agent = DQNAgent(state_dim=state_dim, actions=actions)

    model_path = f"{symbol}_dqn_model.pth"

    if os.path.exists(model_path):
        print("Loading saved model...")
        agent.load_model(model_path)
    else:
        print("Training model...")
        env = TradingEnv(train_data)
        num_episodes = 50
        rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)

                # Safely convert reward to float
                if isinstance(reward, pd.Series):
                    reward = float(reward.iloc[0])
                else:
                    reward = float(reward)

                agent.remember(state, action, reward, next_state if next_state is not None else np.zeros(state_dim), done)
                agent.learn()
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f}")

        agent.save(model_path)
        plot_rewards(rewards)

    print("Testing on unseen 2023 data...")
    env_test = TradingEnv(test_data)
    state = env_test.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env_test.step(action)
        state = next_state

    final_value = env_test.total_value
    if isinstance(final_value, pd.Series):
        final_value = final_value.iloc[-1]

    print(f"Final portfolio value: ${final_value:.2f}")
    plot_trades(test_data, env_test.trades)
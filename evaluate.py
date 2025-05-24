import matplotlib.pyplot as plt

def plot_trades(data, trades, title="Buy/Sell Points on Price Chart"):
    plt.figure(figsize=(14,7))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")

    buys = [t for t in trades if t[2] == "buy"]
    sells = [t for t in trades if t[2] == "sell"]

    if buys:
        plt.scatter([data.index[i] for i, _, _ in buys], [p for _, p, _ in buys], marker='^', color='g', label='Buy', s=100)
    if sells:
        plt.scatter([data.index[i] for i, _, _ in sells], [p for _, p, _ in sells], marker='v', color='r', label='Sell', s=100)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

def plot_rewards(rewards, title="Total Reward per Episode"):
    plt.figure(figsize=(10,5))
    plt.plot(rewards, marker='o')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()
import numpy as np

class TradingEnv:
    def __init__(self, data):
        # Reset index with drop=True to get a clean integer index
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        self.current_step = 0

        self.position = None  # "long" or None
        self.buy_price = 0.0

        self.cash = 10000  # starting cash in $
        self.shares_held = 0
        self.total_value = self.cash

        self.actions = ["buy", "hold", "sell"]

        self.history = []  # to track portfolio value
        self.trades = []   # to track buy/sell points for visualization

    def reset(self):
        self.current_step = 0
        self.position = None
        self.buy_price = 0.0
        self.cash = 10000
        self.shares_held = 0
        self.total_value = self.cash
        self.history = []
        self.trades = []
        return self._get_state()

    def _get_state(self):
        if self.current_step >= self.n_steps:
            return None  # No more data
        row = self.data.iloc[self.current_step]
        state = np.array([
            row["Close"],
            row["MA10"],
            row["MA50"],
            row["RSI"],
            row["Volume"],
            row["Open"]
        ], dtype=np.float32)
        return state

    def step(self, action):
        done = False
        reward = 0.0
        price = self.data.iloc[self.current_step]["Close"]

        # Execute action
        if action == "buy" and self.position is None:
            self.position = "long"
            self.buy_price = price
            self.shares_held = self.cash // price
            self.cash -= self.shares_held * price
            self.trades.append((self.current_step, price, "buy"))

        elif action == "sell" and self.position == "long":
            sell_price = price
            reward = (sell_price - self.buy_price) * self.shares_held
            self.cash += self.shares_held * sell_price
            self.shares_held = 0
            self.position = None
            self.trades.append((self.current_step, price, "sell"))

        else:
            reward = 0.0

        self.total_value = self.cash + self.shares_held * price
        self.history.append(self.total_value)

        self.current_step += 1
        if self.current_step >= self.n_steps:
            done = True

        next_state = self._get_state() if not done else None

        return next_state, reward, done
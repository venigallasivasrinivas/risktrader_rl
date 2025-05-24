import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.q_table.get((state_key, a), 0) for a in self.actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(max_actions)

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        old_q = self.q_table.get((state_key, action), 0)
        next_max_q = max([self.q_table.get((next_state_key, a), 0) for a in self.actions])
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state_key, action)] = new_q
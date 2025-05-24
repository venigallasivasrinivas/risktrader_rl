import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_dim, actions, model_path="models/dqn_model.pth"):
        self.state_dim = state_dim
        self.actions = actions
        self.n_actions = len(actions)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, self.n_actions).to(self.device)
        self.target_net = DQN(state_dim, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.model_path = model_path
        self.update_target_network()
        self.load_model(self.model_path)

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(np.array(state).flatten()).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return self.actions[torch.argmax(q_values).item()]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state), action, reward, np.array(next_state), done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([s[0].flatten() for s in batch])
        actions = np.array([self.actions.index(s[1]) for s in batch])
        rewards = np.array([s[2] for s in batch], dtype=np.float32)
        next_states = np.array([s[3].flatten() for s in batch])
        dones = np.array([s[4] for s in batch], dtype=np.float32)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states_tensor).gather(1, actions_tensor)
        next_q_values = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1).detach()
        expected_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # New save method - compatible with your train.py call: agent.save(path)
    def save(self, path=None):
        path = path or self.model_path
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    # Updated load method with optional path argument
    def load_model(self, path=None):
        path = path or self.model_path
        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.update_target_network()
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print("No saved model found. Training from scratch.")
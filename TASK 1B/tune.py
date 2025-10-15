import numpy as np
import random
import pickle
import os


class QLearningController:
    def __init__(self, n_states=0, n_actions=0, filename="q_table.pkl"):
        """
        Initialize the Q-learning controller.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.filename = filename

        # Learning parameters
        self.lr = 0.1       # learning rate
        self.gamma = 0.9    # discount factor
        self.epsilon = 0.2  # exploration rate

        # Q-table
        self.q_table = np.zeros((n_states, n_actions))

        # Action list
        self.action_list = ["forward", "left", "right", "sharp_left", "sharp_right"]
        self.actions = {}

    # =====================================================
    # Line follower logic (for white line on black background)
    # =====================================================
    def Get_state(self, sensor_data):
        """
        Convert sensor readings [left, center, right] (1=white, 0=black) to state ID.
        """
        left, center, right = sensor_data
        state = left * 4 + center * 2 + right * 1
        return state

    def Calculate_reward(self, state):
        """
        Reward design for white-line following.
        """
        if state == 0b010:
            reward = 10
        elif state in [0b110, 0b011]:
            reward = 5
        elif state in [0b100, 0b001]:
            reward = -2
        else:
            reward = -10
        return reward

    def update_q_table(self, state, action, reward, next_state):
        """
        Q-learning update rule.
        """
        a_idx = self.action_list.index(action)
        current_q = self.q_table[state][a_idx]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][a_idx] = new_q

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_list)
        else:
            best_idx = np.argmax(self.q_table[state])
            return self.action_list[best_idx]

    def perform_action(self, action):
        """
        Map actions to wheel speeds.
        """
        if action == "forward":
            left_speed, right_speed = 2.0, 2.0
        elif action == "left":
            left_speed, right_speed = 1.0, 2.0
        elif action == "right":
            left_speed, right_speed = 2.0, 1.0
        elif action == "sharp_left":
            left_speed, right_speed = 0.5, 2.0
        elif action == "sharp_right":
            left_speed, right_speed = 2.0, 0.5
        else:
            left_speed, right_speed = 0.0, 0.0
        return left_speed, right_speed

    # =====================================================
    # Q-table persistence
    # =====================================================
    def save_q_table(self):
        with open(self.filename, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "n_actions": self.n_actions,
                "n_states": self.n_states
            }, f)

    def load_q_table(self):
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                data = pickle.load(f)
            self.q_table = data.get("q_table", self.q_table)
            self.epsilon = data.get("epsilon", self.epsilon)
            self.n_actions = data.get("n_actions", self.n_actions)
            self.n_states = data.get("n_states", self.n_states)
            return True
        return False


# Optional test block (safe to remove in deployment)
if __name__ == "__main__":
    agent = QLearningController(n_states=8, n_actions=5)
    print("✅ tune.py loaded successfully — no indentation errors!")

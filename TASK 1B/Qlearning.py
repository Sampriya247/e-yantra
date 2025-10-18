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
        self.lr = 0.44      # learning rate
        self.gamma = 0.89   # discount factor
        self.epsilon = 0.3  # exploration rate

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
        lc = 1 if sensor_data['left_corner'] > 0.3 else 0
        l  = 1 if sensor_data['left'] > 0.3 else 0
        m  = 1 if sensor_data['middle'] > 0.3 else 0
        r  = 1 if sensor_data['right'] > 0.3 else 0
        rc = 1 if sensor_data['right_corner'] > 0.3 else 0

    # Combine into binary representation (5 bits)
        state = (lc << 4) | (l << 3) | (m << 2) | (r << 1) | rc
        return state

    def Calculate_reward(self, state):
        """
        Reward design for white-line following.
        """
        if state in [0b00100, 0b01100, 0b00110]:   # centered
            return 20
        elif state in [0b11100, 0b01110, 0b00111]: # near curve
            return 10
        elif state in [0b00010, 0b00001, 0b10000]: # edge
            return -5
        else:
            return -20

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
        if action == 0:    # forward
            return 2.0, 2.0
        elif action == 1:  # left
            return 1.0, 2.5
        elif action == 2:  # right
            return 2.5, 1.0
        elif action == 3:  # sharp left
            return 0.5, 2.5
        elif action == 4:  # sharp right
            return 2.5, 0.5
        else:
            left_speed = 0.0
            right_speed = 0.0

    
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

'''
*
*   ===================================================
*       CropDrop Bot (CB) Theme [eYRC 2025-26]
*   ===================================================
*
*  This script is intended to be an Boilerplate for 
*  Task 1B of CropDrop Bot (CB) Theme [eYRC 2025-26].
*
*  Filename:		Qlearning.py
*  Created:		    24/08/2025
*  Last Modified:	24/08/2025
*  Author:		    e-Yantra Team
*  Team ID:		    [ CB_2202 ]
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using ICT (NMEICT)
*
*****************************************************************************************
'''
'''You can Modify the this file,add more functions According to your usage.
   You are not allowed to add any external packges,Beside the included Packages.You can use Built-in Python modules.
'''
import numpy as np
import random
import pickle
import os

class QLearningController:
    def __init__(self, n_states=0, n_actions=0, filename="q_table.pkl"): 
        """
        Initialize the Q-learning controller.

        Parameters:
        - n_states (int): Total number of discrete states the agent can be in.
        - n_actions (int): Total number of discrete actions the agent can take.
        - filename (str): Filename to save/load the Q-table (persistent learning).
        """

        self.n_states = n_states
        self.n_actions = n_actions

        # === Configure your learning rate (alpha) and exploration rate (epsilon) here ===
        self.lr = 0  # Learning rate: how much new information overrides old
        self.epsilon = 0  # Exploration rate: chance of choosing a random action

        self.filename = filename

        # Initialize Q-table with zeros; dimensions: [states x actions]
        self.q_table = np.zeros((n_states, n_actions))

        # Action list: should be populated with your lineFollowers valid actions.The Below is just an Example.
        self.action_list = [0, 1, 2]  # Example: 0 = left, 1 = forward, 2 = right

        # Mapping of action index to specific commands (e.g., motor speeds)
        self.actions = {}

   import random
import numpy as np

class QLearningAgent:
    def __init__(self, action_list, num_states, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.action_list = action_list
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, len(action_list)))

    def Get_state(self, sensor_data):
        """
        Convert sensor readings (e.g. line sensors) to discrete states.
        Suppose sensor_data = [left, center, right], 0 = white, 1 = black.
        """
        left, center, right = sensor_data

        # Convert pattern of sensors to a unique state ID
        state = left * 4 + center * 2 + right * 1  # binary to decimal
        return state

    def Calculate_reward(self, state):
        """
        Reward function:
        - Center sensor detects line => +10 (ideal)
        - Slightly off-center (left or right) => +5
        - All sensors off line => -10 (punish)
        """
        if state == 0b010:  # Center on line
            reward = 10
        elif state in [0b110, 0b011]:  # Slightly off
            reward = 5
        else:
            reward = -10
        return reward

    def update_q_table(self, state, action, reward, next_state):
        """
        Standard Q-learning update rule:
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s', :)) - Q(s,a))
        """
        a_idx = self.action_list.index(action)
        max_future_q = np.max(self.q_table[next_state])

        current_q = self.q_table[state][a_idx]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][a_idx] = new_q

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore
            return random.choice(self.action_list)
        else:
            # Exploit
            best_action_idx = np.argmax(self.q_table[state])
            return self.action_list[best_action_idx]

    def perform_action(self, action):
        """
        Map each action to differential wheel speeds.
        """
        if action == "forward":
            left_speed, right_speed = 2, 2
        elif action == "left":
            left_speed, right_speed = 1, 2
        elif action == "right":
            left_speed, right_speed = 2, 1
        elif action == "sharp_left":
            left_speed, right_speed = 0.5, 2
        elif action == "sharp_right":
            left_speed, right_speed = 2, 0.5
        else:
            left_speed, right_speed = 0, 0  # stop/fallback
        return left_speed, right_speed


    def save_q_table(self):
        """
        Save the current Q-table and parameters to a file.

        Useful for keeping learned behavior between runs.

        === INSTRUCTIONS: You may Save Additional Thing while saving but do not Remove the the following Parameters ===
        """
        with open(self.filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'n_action': self.n_actions,
                'n_states': self.n_states
                # Add any additional data you want to save
            }, f)

    def load_q_table(self):
        """
        Load the Q-table and parameters from file, if it exists.

        Returns:
        - True if data was loaded successfully, False otherwise.
        """
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data.get('q_table', self.q_table)
            self.epsilon = data.get('epsilon', self.epsilon)
            self.n_actions = data.get('n_action', self.n_actions)
            self.n_states = data.get('n_states', self.n_states)
            # Load other data here if needed
            return True
        return False

'''
*
*   ===================================================
*       CropDrop Bot (CB) Theme [eYRC 2025-26]
*   ===================================================
*
*  This script is intended to be an Boilerplate for 
*  Task 1B of CropDrop Bot (CB) Theme [eYRC 2025-26].
*
*  Filename:		Train.py
*  Created:		    24/08/2025
*  Last Modified:	24/08/2025
*  Author:		    e-Yantra Team
*  Team ID:		    [ CB_XXXX ]
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
   You are not allowed to add any external packges,Beside the included Packages.You can use Built-in Python modules.'''
import time
import signal
import sys

# Import required modules for communication and Q-learning
from Connector import CoppeliaClient       
from Qlearning import QLearningController 

# Flag to handle graceful shutdown when Ctrl+C is pressed
stop_requested = False

def signal_handler(sig, frame):
    """
    Signal handler for keyboard interrupt (Ctrl+C).
    It sets the stop_requested flag to exit the training loop safely.
    """
    global stop_requested
    print("\n[TRAIN] Interrupt received. Stopping training gracefully...")
    stop_requested = True

# Register the signal handler to handle SIGINT
signal.signal(signal.SIGINT, signal_handler)

#=== Add Functions Here ===

def main():
    """
    Main training loop:
    - Initializes Q-learning agent and simulation client.
    - Continuously reads sensor data, updates Q-table based on experience.
    - Saves the Q-table every few iterations and on exit.
    """
    global stop_requested

    # === Q-table & Training Configuration ===
    # Adjust According to Your Need.
    N_STATES = 0         # Number of discrete states (MUST MATCH your Get_state logic)
    N_ACTIONS = 3        # Number of actions available (must match your action_list)
    #Add Other Parameter According to your logic.
    SAVE_INTERVAL = 100  # Save Q-table to disk every N iterations

    # === Initialize Q-learning Controller ===
    ql = QLearningController(n_states=N_STATES, n_actions=N_ACTIONS)

    # Load existing Q-table if it exists (resumes training from last session)
    ql.load_q_table()

    # === Connect to the CoppeliaSim simulator ===
    client = CoppeliaClient()
    client.connect()

    # === Training Loop Initialization ===

    iteration = 0             # Counts training iterations
    prev_state = None         # State before taking an action
    prev_action = None        # Action taken from prev_state

    # === Add Vriables Here ===

    print("[TRAIN] Starting training loop...")

    # === Training Loop Starts Here ===
    while not stop_requested:
        #  Read sensor data from simulator
        sensor_data = client.receive_sensor_data()
        print(sensor_data)  # Optional: Log raw sensor input
        if not sensor_data:
            time.sleep(0.05)
            continue  # Skip iteration if sensor data is invalid

        # Convert sensor data to a discrete state
        state = ql.Get_state(sensor_data)

        # If not the first iteration, update Q-table using previous state and action
        if prev_state is not None and prev_action is not None:
            reward = ql.Calculate_reward(state)  # Compute reward for action taken
            ql.update_q_table(prev_state, prev_action, reward, state)  # Q-learning update

        # Choose next action based on current state (explore or exploit)
        action = ql.choose_action(state)

        # Convert action into motor speeds (left, right)
        left_speed, right_speed = ql.perform_action(action)

        # Send motor command to the simulator
        client.send_motor_command(left_speed, right_speed, state=state, action=action,reward=reward)

        prev_state = state
        prev_action = action

        iteration += 1  # Increment training step count

        # Save Q-table periodically
        if iteration % SAVE_INTERVAL == 0:
            ql.save_q_table()
            print(f"[TRAIN] Saved Q-table at iteration {iteration}")

        # Control loop timing 
        time.sleep(0.05)

    # === When training is interrupted (Ctrl+C) ===
    ql.save_q_table()  # Save final Q-table before exiting
    client.send_motor_command(0, 0)  # Stop the robot
    client.close()  # Disconnect from simulator
    print("[TRAIN] Training stopped and Q-table saved.")


# Script entry point
if __name__ == "__main__":
    main()

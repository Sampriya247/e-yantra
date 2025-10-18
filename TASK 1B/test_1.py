'''
*
*   ===================================================
*       CropDrop Bot (CB) Theme [eYRC 2025-26]
*   ===================================================
*
*  This script tests the trained Q-learning agent to
*  follow a white line on a black background using
*  the saved Q-table (no further learning).
*
*  Filename:        Test.py
*  Created:         24/08/2025
*  Last Modified:   18/10/2025
*  Author:          e-Yantra Team (Modified by Sampriya)
*  Team ID:         [ CB_XXXX ]
*
*****************************************************************************************
'''

import time
import signal
import sys

# Import required modules
from Connector import CoppeliaClient
from Qlearning import QLearningController   # Ensure filename matches exactly (Qlearning.py)

# Global flag for safe interruption
stop_requested = False


def signal_handler(sig, frame):
    """
    Handle Ctrl+C to stop the test loop safely.
    """
    global stop_requested
    print("\n[TEST] Interrupt received. Stopping the test loop gracefully...")
    stop_requested = True


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def main():
    global stop_requested

    # ============================================
    # Initialize Q-learning controller
    # ============================================
    ql = QLearningController(n_states=8, n_actions=5)  # Must match Train.py setup
    loaded = ql.load_q_table()

    if not loaded:
        print("[TEST] ❌ No saved Q-table found (q_table.pkl). Train first using Train.py.")
        return
    else:
        print("[TEST] ✅ Q-table loaded successfully. Starting test run...")

    # Force pure exploitation — disable exploration
    ql.epsilon = 0.0

    # ============================================
    # Connect to simulator
    # ============================================
    client = CoppeliaClient()
    client.connect()
    print("[TEST] Connected to CoppeliaSim successfully.")

    # ============================================
    # Test loop
    # ============================================
    print("[TEST] Running line-following using trained policy...")
    time.sleep(1.0)  # Small delay before starting

    while not stop_requested:
        # Step 1: Get sensor data
        sensor_data = client.receive_sensor_data()
        if not sensor_data or len(sensor_data) != 3:
            time.sleep(0.05)
            continue

        # Step 2: Determine the current state
        state = ql.Get_state(sensor_data)

        # Step 3: Choose best (greedy) action
        action = ql.choose_action(state)

        # Step 4: Convert action to motor speeds
        left_speed, right_speed = ql.perform_action(action)

        # Step 5: Send command to simulator
        client.send_motor_command(left_speed, right_speed, state=state, action=action)

        # Optional: Log every few seconds
        print(f"[TEST] Sensors: {sensor_data} | State: {state} | Action: {action}")

        # Step 6: Small control delay (tune if needed)
        time.sleep(0.05)

    # ============================================
    # Safe exit
    # ============================================
    client.send_motor_command(0, 0)
    client.close()
    print("[TEST] 🚦 Testing stopped. Bot halted safely.")


# Entry point
if __name__ == "__main__":
    main()

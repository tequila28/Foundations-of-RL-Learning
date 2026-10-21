#!/bin/bash
# Specifies the interpreter to be used for executing the script: the Bash shell located at /bin/bash.

# GridWorld parameters configuration

SIZE=5 # Dimension of the square grid world (number of rows and columns). Here, it creates a 5x5 grid.

GAMMA=0.9 # Discount factor for future rewards in reinforcement learning. A value between 0 and 1, where values closer to 1 prioritize future rewards more highly.

ACTIONS="up right down left stay" # Set of possible actions the agent can take. 'stay' is an action that results in remaining in the current grid cell.

FORBIDDEN_STATES="6 7 12 16 18 21" # List of state indices (cell numbers) that are blocked or hazardous. The agent will receive the penalty R_FORBID upon entering any of these states.

TARGET_STATES="17" # List of state indices representing goal states. The agent will receive the reward R_TARGET upon reaching any of these states.

R_BOUND=-1 # Immediate reward received when the agent's action would move it outside the grid boundary (hits a wall). This is a penalty to discourage invalid moves.

R_FORBID=-1 # Immediate reward received when the agent enters a forbidden state. This is a penalty to discourage entering hazardous areas.

R_TARGET=1 # Immediate reward received when the agent reaches a target state. This is a positive reward for achieving the objective.

R_DEFAULT=0 # Default immediate reward received for any other valid transition (moving to a non-target, non-forbidden cell).

SEED=42 # Random seed value used to initialize the pseudorandom number generator. This ensures reproducibility of experimental results.

# Execute the main program, passing all configured parameters as command-line arguments
python Chapter1_Basic_Concepts/src/experiment.py \
    --size $SIZE \
    --gamma $GAMMA \
    --actions $ACTIONS \
    --forbidden_states $FORBIDDEN_STATES \
    --target_states $TARGET_STATES \
    --r_bound $R_BOUND \
    --r_forbid $R_FORBID \
    --r_target $R_TARGET \
    --r_default $R_DEFAULT \
    --seed $SEED
#!/bin/bash

# GridWorld Parameters Configuration
SIZE=5                           # Grid size (5x5)
GAMMA=0.9                        # Discount factor
FORBIDDEN_STATES="6 7 12 16 18 21"  # Forbidden states (cannot enter)
TARGET_STATES="17"               # Target/terminal states
R_BOUND=-1                       # Reward for hitting boundary
R_FORBID=-1                      # Reward for entering forbidden states
R_TARGET=1                       # Reward for reaching target states
R_DEFAULT=0                      # Default reward for normal movement

# Algorithm Parameters
THETA=1e-6                       # Convergence threshold
MAX_ITERATION=1000               # Maximum number of iterations

TRUNCATED_INNER_ITERATION=5    # Default inner iteration count for truncated policy iteration

TRUNCATED_INNER_ITERATION_LIST="1 5 10 20 50 100"  # List of inner iterations for sensitivity analysis

# Run the main program
python Chapter3_Policy_and_Value_Iteration/src/experiment.py \
    --size $SIZE \
    --gamma $GAMMA \
    --forbidden_states $FORBIDDEN_STATES \
    --target_states $TARGET_STATES \
    --r_bound $R_BOUND \
    --r_forbid $R_FORBID \
    --r_target $R_TARGET \
    --r_default $R_DEFAULT \
    --theta $THETA \
    --max_iteration $MAX_ITERATION \
    --truncated_inner_iteration $TRUNCATED_INNER_ITERATION \
    --truncated_inner_iteration_list $TRUNCATED_INNER_ITERATION_LIST \
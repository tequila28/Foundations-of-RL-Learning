#!/bin/bash

# GridWorld configuration parameters
SIZE=5
GAMMA=0.9
FORBIDDEN_STATES="6 7 12 16 18 21"
TARGET_STATES="17"
R_BOUND=-1
R_FORBID=-1
R_TARGET=1
R_DEFAULT=0

# Temporal Difference algorithm training parameters
NUM_EPISODES=100
MAX_STEPS=1000

# Algorithm hyperparameters
LEARNING_RATE=0.1
EPSILON=0.2
EPSILON_DECAY=0.998
EPSILON_MIN=0.01

# N-step SARSA parameters
N_STEPS="1 10"


# Run Temporal Difference algorithm comparison experiment
python Chapter6_Temporal_Difference/src/experiment.py \
    --size $SIZE \
    --gamma $GAMMA \
    --forbidden_states $FORBIDDEN_STATES \
    --target_states $TARGET_STATES \
    --r_bound $R_BOUND \
    --r_forbid $R_FORBID \
    --r_target $R_TARGET \
    --r_default $R_DEFAULT \
    --num_episodes $NUM_EPISODES \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --epsilon $EPSILON \
    --epsilon_decay $EPSILON_DECAY \
    --epsilon_min $EPSILON_MIN \
    --n_steps $N_STEPS \
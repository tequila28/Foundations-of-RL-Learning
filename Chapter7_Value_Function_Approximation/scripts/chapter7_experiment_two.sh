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

# Training parameters
NUM_EPISODES=1000
MAX_STEPS=100

# Algorithm hyperparameters
LEARNING_RATE=0.0005
EPSILON=0.2
EPSILON_DECAY=0.998
EPSILON_MIN=0.01



python Chapter7_Value_Function_Approximation/src/experiment_two/experiment.py \
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
    --epsilon_min $EPSILON_MIN
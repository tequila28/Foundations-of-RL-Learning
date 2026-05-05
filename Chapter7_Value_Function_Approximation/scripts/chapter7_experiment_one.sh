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

N_EPISODES=500
LEARNING_RATE=0.001
SEED=42



python Chapter7_Value_Function_Approximation/src/experiment_one/experiment.py \
    --size $SIZE \
    --gamma $GAMMA \
    --forbidden_states $FORBIDDEN_STATES \
    --target_states $TARGET_STATES \
    --r_bound $R_BOUND \
    --r_forbid $R_FORBID \
    --r_target $R_TARGET \
    --r_default $R_DEFAULT \
    --n_episodes $N_EPISODES \
    --learning_rate $LEARNING_RATE \
    --seed $SEED
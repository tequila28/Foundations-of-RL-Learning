#!/bin/bash

# GridWorld configuration parameters
SIZE=5
GAMMA=0.9
FORBIDDEN_STATES="6 7 12 16 18 21"
TARGET_STATES="17"
R_BOUND=-1
R_FORBID=-1
R_TARGET=10
R_DEFAULT=0

# Policy Gradient algorithm training parameters
NUM_EPISODES=5000
MAX_STEPS=20
HIDDEN_SIZE=128
LEARNING_RATE=0.001
SEED=42
FEATURE_TYPE="one_hot"

# Actor-Critic specific parameters
GAE_LAMBDA=0.95

# Run Policy Gradient algorithm comparison experiment
python Chapter8_Policy_Gradient/src/experiment.py \
    --size $SIZE \
    --gamma $GAMMA \
    --forbidden_states $FORBIDDEN_STATES \
    --target_states $TARGET_STATES \
    --r_bound $R_BOUND \
    --r_forbid $R_FORBID \
    --r_target $R_TARGET \
    --r_default $R_DEFAULT \
    --n_episodes $NUM_EPISODES \
    --max_steps $MAX_STEPS \
    --hidden_size $HIDDEN_SIZE \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --feature_type $FEATURE_TYPE \
    --gae_lambda $GAE_LAMBDA
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

# Monte Carlo algorithm configuration parameters
# MC Basic parameters
MC_BASIC_EPISODE_LENGTH=10
MC_BASIC_ITERATIONS=100

# MC Exploring Starts parameters
MC_ES_EPISODE_LENGTH=10
MC_ES_ITERATIONS=10000

# MC ε-greedy (first) parameters
MC_EPS1_EPISODE_LENGTH=1000
MC_EPS1_EPSILON=0.1
MC_EPS1_ITERATIONS=100

# MC ε-greedy (second) parameters
MC_EPS2_EPISODE_LENGTH=1000
MC_EPS2_EPSILON=0.2
MC_EPS2_ITERATIONS=100

# Scatter plot parameters
MC_EPISODE_LENGTH=100000
MC_EPS_EPSILON1=1.0
MC_EPS_EPSILON2=0.2

# Run Monte Carlo algorithm comparison experiment
python Chapter4_Monte_Carlo/src/experiment.py \
    --size $SIZE \
    --gamma $GAMMA \
    --forbidden_states $FORBIDDEN_STATES \
    --target_states $TARGET_STATES \
    --r_bound $R_BOUND \
    --r_forbid $R_FORBID \
    --r_target $R_TARGET \
    --r_default $R_DEFAULT \
    --mc_basic_episode_length $MC_BASIC_EPISODE_LENGTH \
    --mc_basic_iterations $MC_BASIC_ITERATIONS \
    --mc_es_episode_length $MC_ES_EPISODE_LENGTH \
    --mc_es_iterations $MC_ES_ITERATIONS \
    --mc_eps1_episode_length $MC_EPS1_EPISODE_LENGTH \
    --mc_eps1_epsilon $MC_EPS1_EPSILON \
    --mc_eps1_iterations $MC_EPS1_ITERATIONS \
    --mc_eps2_episode_length $MC_EPS2_EPISODE_LENGTH \
    --mc_eps2_epsilon $MC_EPS2_EPSILON \
    --mc_eps2_iterations $MC_EPS2_ITERATIONS \
    --mc_episode_length $MC_EPISODE_LENGTH \
    --mc_eps_epsilon1 $MC_EPS_EPSILON1 \
    --mc_eps_epsilon2 $MC_EPS_EPSILON2
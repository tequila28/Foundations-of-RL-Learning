#!/bin/bash

# Default parameters
# Size of the square for uniform distribution
SQUARE_SIZE=50.0
# Number of samples to generate
NUM_SAMPLES=1000
# Total number of iterations for optimization

TOTAL_ITERATIONS=400

# Initial x-coordinate
INIT_X=50.0
# Initial y-coordinate
INIT_Y=50.0

# Batch size(s) for mini-batch gradient descent (can specify multiple)
BATCH_SIZES="10 50"
# Constant learning rate (used when alpha_type is 'constant')
CONSTANT_ALPHA=0.005

# Run mean estimation experiment
python Chapter5_Stochastic_Approximation/src/experiment.py \
    --square_size $SQUARE_SIZE \
    --num_samples $NUM_SAMPLES \
    --total_iterations $TOTAL_ITERATIONS \
    --init_x $INIT_X \
    --init_y $INIT_Y \
    --batch_size $BATCH_SIZES \
    --constant_alpha $CONSTANT_ALPHA
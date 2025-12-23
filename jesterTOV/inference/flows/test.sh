#!/bin/bash

# For testing, this is with low number of samples and epochs
# Using masked autoregressive flow with RationalQuadraticSpline transformer

uv run python -m jesterTOV.inference.flows.train_flow \
    --posterior-file ../data/gw170817/gw170817_gwtc1_lowspin_posterior.npz \
    --output-dir ./models/test/ \
    --num-epochs 3000 \
    --learning-rate 1e-4 \
    --max-patience 200 \
    --flow-type masked_autoregressive_flow \
    --transformer rational_quadratic_spline \
    --transformer-knots 8 \
    --transformer-interval 4 \
    --nn-depth 4 \
    --batch-size 64 \
    --max-samples 20000 \
    --standardize \
    --plot-corner \
    --plot-losses
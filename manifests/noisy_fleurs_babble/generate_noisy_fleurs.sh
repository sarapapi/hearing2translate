#!/bin/bash

# Run noisy FLEURS generation script
python generate_noisy_fleurs.py \
    --fleurs-manifest-dir "hearing2translate/manifests/fleurs" \
    --manifest-dir "hearing2translate/manifests" \
    --h2t-datadir "hearing2translate/data" \
    --musan-path "musan" \
    --noise-type "babble" # or ambient

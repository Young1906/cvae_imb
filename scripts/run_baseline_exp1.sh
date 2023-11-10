#!/bin/bash
CONFIGS="config/exp-baseline/*"

for f in $CONFIGS
do
	python -m modules.baseline.exp -C $f
done

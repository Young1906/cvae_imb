#!/bin/bash
CONFIGS="config/exp-baseline/*"

for f in $CONFIGS
do
	for i in {1..15}
	do
		python -m modules.baseline.exp -C $f
	done
done

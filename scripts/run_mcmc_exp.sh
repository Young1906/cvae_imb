#!/bin/bash
CONFIGS="config/exp/*"

for f in $CONFIGS
do
	for i in {1..15}
	do
		python -m modules.mcmc.exp -C $f
	done
done

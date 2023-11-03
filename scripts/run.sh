#!/bin/bash
CONFIGS="config/mcmc/*"

for f in $CONFIGS
do
	for i in {1..15}
	do
		python -m modules.mcmc -C $f
	done
done

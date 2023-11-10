#!/bin/bash
CONFIGS="config/exp-baseline1/*"

for f in $CONFIGS
do
	python -m modules.baseline.exp -C $f
done

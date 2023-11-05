
#!/bin/bash
CONFIGS="config/baseline/*"

for f in $CONFIGS
do
	for i in $(seq 1 15)
	do
		echo "Experiment: ${f}, trial ${i}"
		timeout 120 python -m modules.baseline -C $f
	done
done

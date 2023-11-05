
#!/bin/bash
CONFIGS="config/baseline/*"

for f in $CONFIGS
do
	timeout 300 python -m modules.baseline -C $f
done

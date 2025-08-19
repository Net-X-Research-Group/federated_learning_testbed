#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <NUM OF NODES> <DISTRIBUTION>"
    exit 1
fi

N=$1
DIST=$2

for i in $(seq 1 20);
do
	helper_scripts/dataset_distributor.sh "${N}" "${DIST}"
	mv cifar10_"${N}"_partitions_"${DIST}".png cifar10_"${N}"_partitions_"${DIST}"_trial_"${i}".png
	sleep 2
	flwr run flwr-application --stream
	sleep 2
	#kill -9 $(pgrep tshark)
  helper_scripts/transfer_latency_measurements.sh "${N}"
done

flwr ls flwr-application --format json > trials.json

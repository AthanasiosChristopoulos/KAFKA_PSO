#!/bin/bash

cd ./java || exit 1

set -a
source .env
set +a

mvn -q -e -DskipTests \
  -Dexec.mainClass=pso.WorkerMain \
  -Dexec.args="1" \
  -P"$ND4J_PROFILE" \
  compile exec:java
#!/bin/bash

cd ../java
# mvn -q -DskipTests clean  
# mvn -q -P$ND4J_PROFILE clean compile

# ==============================================================

set -a     
source .env
set +a
# sleep 3s

mvn -q -e -DskipTests \
    -Dexec.mainClass=pso.CoordinatorMain \
    -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
    -P$ND4J_PROFILE compile exec:java
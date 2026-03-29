#!/bin/bash

cd ./java
# mvn -q -DskipTests clean  
# mvn -q -P$ND4J_PROFILE clean compile

# ==============================================================

set -a     
source .env
set +a

mvn -q -e -DskipTests \
-Dexec.mainClass=pso.CoordinatorMain \
 -P$ND4J_PROFILE compile exec:java
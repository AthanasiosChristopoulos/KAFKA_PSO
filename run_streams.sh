#!/bin/bash

cd ./java
# mvn -q -DskipTests clean  
# mvn -q -P$ND4J_PROFILE clean compile

# ==============================================================

set -a     
source .env
set +a
  
# ==============================================================
# Less debugging messages:

mvn -q -e -DskipTests \
-Dexec.mainClass=experimentation.Experimentation \
 -P$ND4J_PROFILE compile exec:java

# ==============================================================

mvn -q -e -DskipTests \
-Dexec.mainClass=experimentation.Experimentation \
-Dorg.slf4j.simpleLogger.defaultLogLevel=debug \
-Dorg.slf4j.simpleLogger.log.org.deeplearning4j=debug \
-Dorg.slf4j.simpleLogger.log.org.nd4j=debug \
 -P$ND4J_PROFILE compile exec:java

# ==============================================================

curl -d "hello from $(hostname)" https://ntfy.sh/pso-test-123   # sends a notification to this site when training has finished, which makes a notification noise
#!/bin/bash

cd ./java

delete=2
set -a           # auto-export all variables
source .env
set +a

# mvn -q -e -DskipTests \
# -Dexec.mainClass=experimentation.Experimentation \
# -Dorg.slf4j.simpleLogger.defaultLogLevel=debug \
# -Dorg.slf4j.simpleLogger.log.org.deeplearning4j=debug \
# -Dorg.slf4j.simpleLogger.log.org.nd4j=debug \
#  -P$ND4J_PROFILE compile exec:java

printf '\a'
#!/bin/bash

cd ../java

set -a
source .env
set +a

# Run one Worker ==============================================================

# mvn -e -DskipTests \
#   -Dexec.mainClass=pso.WorkerMain \
#   -Dexec.args="0" \
#   -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
#   -P"$ND4J_PROFILE" \
#   compile exec:java
  
# mvn -e -DskipTests \
#   -Dexec.mainClass=pso.WorkerMain \
#   -Dexec.args="0 1 2 3" \
#   -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
#   -Pcpu \
#   compile exec:java

# mvn -e -DskipTests \
#   -Dexec.mainClass=pso.WorkerMain \
#   -Dexec.args="0 1 2 3 4" \
#   -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
#   -Pcpu \
#   compile exec:java

# mvn -e -DskipTests \
#   -Dexec.mainClass=pso.WorkerMain \
#   -Dexec.args="0 1 2 3 4" \
#   -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
#   -Pcpu \
#   compile exec:java


mvn -e -DskipTests \
  -Dexec.mainClass=pso.WorkerMain \
  -Dexec.args="0 1 2 3 4 5 6 7 8 9 10 11" \
  -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
  -Pcpu \
  compile exec:java

# Run one Worker JAR ==============================================================

# Create JAR:
# mvn -DskipTests package
# /usr/bin/time -v java \
#   -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
#   -XX:+UnlockDiagnosticVMOptions \
#   -XX:NativeMemoryTracking=summary \
#   -cp target/pso-streams-1.0.0.jar \
#   pso.WorkerMain 0

# /usr/bin/time -v java \
#   -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
#   -XX:+UnlockDiagnosticVMOptions \
#   -XX:NativeMemoryTracking=summary \
#   -cp target/pso-streams-1.0.0.jar \
#   pso.WorkerMain 0 1 2 3

# Run Multiple Workers ==============================================================

# mvn -q -e -DskipTests \
#   -Dexec.mainClass=pso.WorkerMain \
#   -Dexec.args="0" \
#   -P"$ND4J_PROFILE" \
#   compile exec:java

# mvn -e -DskipTests \
#   -Dexec.mainClass=pso.WorkerGroupMain \
#   -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
#   -Pcpu \
#   compile exec:java
#!/bin/bash

cd ./java
# mvn -q -DskipTests clean  => deletes target

delete=2
set -a           # auto-export all variables
source .env
set +a

if [[ -n "$1" && "$1" != "--reset" && "$1" != "--debug" ]]; then    # if there is an command line argument to the script run_streams, 
                                                                    # and it isnt debug or reset, then its N_WORKERS. Override this variable from .env with the argument  
    export N_WORKERS="$1"
fi

if [[ -n "$1" ]]; then
    echo "$1"
fi

if [[ "$1" != "--debug" ]]; then

    BROKER="broker"
    BOOTSTRAP="localhost:9092"

    if [[ "$FULLY_INFORMED" == "true" || "$ENABLE_NEIGHBORHOODS" == "true" ]]; then
        TOPICS=(
            "$PBEST_WEIGHTS_TOPIC"
        )

    elif [[ "$1" == "--reset" ]]; then
        
        TOPICS=(
            "$PBEST_WEIGHTS_TOPIC"
            "$GLOBAL_WEIGHTS_TOPIC"
            "$LOCAL_WEIGHTS_TOPIC"
        )

    else
        TOPICS=(
            "$GLOBAL_WEIGHTS_TOPIC"
        )
    fi


    for topic in "${TOPICS[@]}"; do
        docker exec -it "$BROKER" /opt/kafka/bin/kafka-topics.sh \
            --bootstrap-server "$BOOTSTRAP" \
            --delete --topic "$topic"
    done

    for topic in "${TOPICS[@]}"; do
        docker exec -it "$BROKER" /opt/kafka/bin/kafka-topics.sh \
            --bootstrap-server "$BOOTSTRAP" \
            --create --topic "$topic" \
            --partitions 1 --if-not-exists  # this 1 could be N_WORKERS so this would work with 
                                            # the exact partitions (1 partition per worker)
    done

    echo Executing ...

fi

export RUN_ID="$(date +%Y%m%d_%H%M%S)"

# Memory Limiters =====================================================
# export MAVEN_OPTS="\
# -Xms512M -Xmx1G \
# -Dorg.bytedeco.javacpp.maxbytes=1000M \
# -Dorg.bytedeco.javacpp.maxphysicalbytes=6G"

# export JAVA_TOOL_OPTIONS="\
# -Xms512M -Xmx1G \
# -Dorg.bytedeco.javacpp.maxbytes=1000M \
# -Dorg.bytedeco.javacpp.maxphysicalbytes=3000M"

# =====================================================================
# mvn -q -DskipTests clean compile exec:java
# mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java

# mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java

# mvn -q -DskipTests package                 
# java -jar target/iris-streams-1.0.0.jar    

# mvn -q -DskipTests -Dexec.mainClass=pso.Simulation -P$ND4J_PROFILE clean compile exec:java  # this ND4J_PROFILE .env variables select pom.xml P = profile

mvn -q -e -DskipTests -Dexec.mainClass=pso.Simulation -P$ND4J_PROFILE compile exec:java  

# mvn -DskipTests -Dexec.mainClass=pso.Simulation -P$ND4J_PROFILE compile exec:java  

# mvn -q -e -DskipTests \
#   -Dexec.mainClass=pso.Simulation \
#   -Dexec.jvmArgs="-Xms512M -Xmx1G -Dorg.bytedeco.javacpp.maxbytes=1000M -Dorg.bytedeco.javacpp.maxphysicalbytes=6G" \
#   -P$ND4J_PROFILE \
#   compile exec:java
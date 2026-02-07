#!/bin/bash

cd ./java
mvn -q -DskipTests clean

delete=2
set -a           # auto-export all variables
source .env
set +a

if [[ -n "$1" && "$1" != "--reset" && "$1" != "--debug" ]]; then    # if there is an command line argument to the script run_streams, 
                                                                    # and it isnt debug or reset, then its N_WORKERS. Override this variable from .env with the argument  
    export N_WORKERS="$1"
fi

# --- Reset section -----------------------------------------------------------

# if [[ "$1" == "--reset" ]]; then    # execute only if there is a reset flag

#     for ((i=0; i<N_WORKERS; i++)); do
#         app="pso-worker-$i"
#         docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
#             --application-id "$app" \
#             --input-topics iris-input \
#             --bootstrap-server localhost:9092 \
#             --force
#     done

#     # app="pso-coordinator"
#     # docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
#     #     --application-id "$app" \
#     #     --input-topics local-weights-topic \
#     #     --bootstrap-server localhost:9092 \
#     #     --force

# fi

if [[ -n "$1" ]]; then
    echo "$1"
fi

if [[ "$1" != "--debug" ]]; then

    BROKER="broker"
    BOOTSTRAP="localhost:9092"

    if [[ "$FULLY_INFORMED" == "true" ]]; then
        echo "Fully Informed run"
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
        echo "Neighborhood Best run"
        TOPICS=(
            # "$PBEST_WEIGHTS_TOPIC"
            "$GLOBAL_WEIGHTS_TOPIC"
            # "$LOCAL_WEIGHTS_TOPIC"
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
            --partitions 1 --if-not-exists
    done

    echo Executing ...

fi

export RUN_ID="$(date +%Y%m%d_%H%M%S)"

# mvn -q -DskipTests clean compile exec:java
# mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java

# mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java

# mvn -q -DskipTests package                 
# java -jar target/iris-streams-1.0.0.jar    

mvn -q -DskipTests -Dexec.mainClass=pso.Simulation -P$ND4J_PROFILE clean compile exec:java  # this ND4J_PROFILE .env variables select pom.xml P = profile






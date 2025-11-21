#!/bin/bash

set -a           # auto-export all variables
source .env
set +a

# --- Reset section -----------------------------------------------------------
if [[ "$1" == "--reset" ]]; then    # execute only if there is a reset flag

    for ((i=0; i<NUM_WORKERS; i++)); do
        app="pso-worker-$i"
        docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
            --application-id "$app" \
            --input-topics iris-input \
            --bootstrap-server localhost:9092 \
            --force
    done

    # app="pso-coordinator"
    # docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
    #     --application-id "$app" \
    #     --input-topics local-weights-topic \
    #     --bootstrap-server localhost:9092 \
    #     --force

fi

BROKER="broker"
BOOTSTRAP="localhost:9092"

if [[ "$FULLY_INFORMED" == "true" ]]; then
    echo "Fully Informed run"
    TOPICS=(
        "$PBEST_WEIGHTS_TOPIC"
    )
else
    TOPICS=(
        "$PBEST_WEIGHTS_TOPIC"
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
        --partitions 1 --if-not-exists
done

echo pause 1 second
sleep 1
echo Awoken

export RUN_ID="$(date +%Y%m%d_%H%M%S)"

# mvn -q -DskipTests clean compile exec:java

mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java

# mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java

# mvn -q -DskipTests package                 
# java -jar target/iris-streams-1.0.0.jar    







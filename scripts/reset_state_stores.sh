#!/bin/bash

set -a           # auto-export all variables
cd ../java
source .env
set +a

# --- Reset section -----------------------------------------------------------

for ((i=0; i<N_WORKERS; i++)); do
    app="pso-worker-$i"
    docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
        --application-id "$app" \
        --input-topics iris-input \
        --bootstrap-server localhost:9092 \
        --force
done

app="pso-coordinator"
docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
    --application-id "$app" \
    --input-topics local-weights-topic \
    --bootstrap-server localhost:9092 \
    --force



echo "$FULLY_INFORMED"
echo "$PBEST_WEIGHTS_TOPIC"
echo "$GLOBAL_WEIGHTS_TOPIC"

if [[ "1" == "2" ]]; then

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
    echo Executing ...

fi





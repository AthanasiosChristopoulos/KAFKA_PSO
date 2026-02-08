#!/bin/bash

BROKER="broker"
BOOTSTRAP="localhost:9092"

TOPICS=(
  "local-weights-topic"
  "global-weights-topic"
  "pbest-weights-topic"
)

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

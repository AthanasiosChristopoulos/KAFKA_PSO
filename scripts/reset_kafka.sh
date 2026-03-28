#!/bin/bash

BROKER="broker"
BOOTSTRAP="localhost:9092"

# ================================================================
# Recreate topics

TOPICS=(
  "local-weights-topic"
  "global-weights-topic"
  "pbest-weights-topic"
)

for topic in "${TOPICS[@]}"; do                                       # Delete topics
    docker exec -it "$BROKER" /opt/kafka/bin/kafka-topics.sh \
        --bootstrap-server "$BOOTSTRAP" \
        --delete --topic "$topic"
done

for topic in "${TOPICS[@]}"; do                                       # Recreate topics
    docker exec -it "$BROKER" /opt/kafka/bin/kafka-topics.sh \
        --bootstrap-server "$BOOTSTRAP" \
        --create --topic "$topic" \
        --partitions 1 --if-not-exists
done

set -euo pipefail

BROKER="broker"
BOOTSTRAP="localhost:9092"

# ================================================================
PREFIX_REGEX="(pso-coordinator|pso-gbest)"
SUFFIX="-gBestEmitStore-changelog"

echo "Searching for topics matching:"
echo "  ${PREFIX_REGEX}*${SUFFIX}"
echo

TOPICS=$(
  docker exec -i "$BROKER" /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server "$BOOTSTRAP" \
    --list \
  | grep -E "^${PREFIX_REGEX}.*${SUFFIX}$" || true
)

if [[ -z "$TOPICS" ]]; then
  echo "No matching topics found."
  exit 0
fi

echo "Found the following topics:"
echo "$TOPICS"
echo

for topic in $TOPICS; do
  echo "Deleting: $topic"
  docker exec -i "$BROKER" /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server "$BOOTSTRAP" \
    --delete \
    --topic "$topic" || true
done

echo
echo "Done deleting changelog topics."

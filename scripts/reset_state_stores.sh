# #!/bin/bash

# set -a           # auto-export all variables
# cd ../java
# source .env
# set +a

# # --- Reset section -----------------------------------------------------------

# for ((i=0; i<N_WORKERS; i++)); do
#     app="pso-worker-$i"
#     docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
#         --application-id "$app" \
#         --input-topics iris-input \
#         --bootstrap-server localhost:9092 \
#         --force
# done

# app="pso-coordinator"
# docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
#     --application-id "$app" \
#     --input-topics local-weights-topic \
#     --bootstrap-server localhost:9092 \
#     --force



# echo "$FULLY_INFORMED"
# echo "$PBEST_WEIGHTS_TOPIC"
# echo "$GLOBAL_WEIGHTS_TOPIC"

# if [[ "1" == "2" ]]; then

#     BROKER="broker"
#     BOOTSTRAP="localhost:9092"

#     if [[ "$FULLY_INFORMED" == "true" ]]; then
#         echo "Fully Informed run"
#         TOPICS=(
#             "$PBEST_WEIGHTS_TOPIC"
#         )
#     else
#         TOPICS=(
#             "$PBEST_WEIGHTS_TOPIC"
#             "$GLOBAL_WEIGHTS_TOPIC"
#         )
#     fi

#     for topic in "${TOPICS[@]}"; do
#         docker exec -it "$BROKER" /opt/kafka/bin/kafka-topics.sh \
#             --bootstrap-server "$BOOTSTRAP" \
#             --delete --topic "$topic"
#     done

#     for topic in "${TOPICS[@]}"; do
#         docker exec -it "$BROKER" /opt/kafka/bin/kafka-topics.sh \
#             --bootstrap-server "$BOOTSTRAP" \
#             --create --topic "$topic" \
#             --partitions 1 --if-not-exists
#     done

#     echo pause 1 second
#     sleep 1
#     echo Executing ...

# fi




#!/bin/bash
set -euo pipefail

BROKER="broker"
BOOTSTRAP="localhost:9092"

APP_PREFIX="pso-coordinator"
SUFFIX="-gBestEmitStore-changelog"

echo "Searching for topics matching:"
echo "  ${APP_PREFIX}*${SUFFIX}"
echo

TOPICS=$(
  docker exec -i "$BROKER" /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server "$BOOTSTRAP" \
    --list \
  | grep "^${APP_PREFIX}.*${SUFFIX}$" || true
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
echo "Done deleting coordinator changelog topics."
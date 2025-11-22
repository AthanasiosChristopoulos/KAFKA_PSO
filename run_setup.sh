#! /bin/bash

# 0) ==============================================================

# docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
#   --bootstrap-server localhost:9092 \
#   --create --topic iris-input --partitions 3 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list 

# Reset StateStores ===================================================================

for app in pso-worker pso-coordinator; do
  docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
    --application-id "$app" \
    --input-topics iris-input \
    --bootstrap-server localhost:9092 \
    --force
done

# DATA_TOPIC ==============================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic iris-input --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic iris-input

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic iris-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-input --from-beginning
  

# PBEST_WEIGHTS_TOPIC ==============================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic pbest-weights-topic --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic pbest-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic pbest-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic pbest-weights-topic --from-beginning

    docker exec -i broker bash -lc '
      /opt/kafka/bin/kafka-console-consumer.sh \
        --bootstrap-server localhost:9092 \
        --topic pbest-weights-topic 
    ' | jq -c '{id_worker, pBestMsgIndex, accuracy}'

# LOCAL_WEIGHTS_TOPIC ==============================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic local-weights-topic --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic local-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic local-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic local-weights-topic --from-beginning

# GLOBAL_WEIGHTS_TOPIC ==============================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic global-weights-topic --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic global-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic global-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic global-weights-topic --from-beginning

# PREDICTION_TOPIC ==============================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic iris-output --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic iris-output

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-output --from-beginning


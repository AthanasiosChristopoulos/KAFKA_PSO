#! /bin/bash

# ==============================================================
# recreate:

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic iris-input --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic pbest-weights-topic --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic local-weights-topic --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic global-weights-topic --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic prediction_input --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic prediction_output --partitions 1 --if-not-exists

cd python
python3 iris_data_producer.py --all

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list 

exit 0

#  ==============================================================

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

# iris: ======================================================
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
  
# iris: ======================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic wine-input --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic wine-input

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic wine-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic wine-input --from-beginning
  

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

# PREDICTION_INPUT_TOPIC ==============================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic prediction_input --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic prediction_input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction_input --from-beginning

# PREDICTION_OUTPUT_TOPIC ==============================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic prediction_output --partitions 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic prediction_output

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction_output --from-beginning

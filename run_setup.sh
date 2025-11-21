#! /bin/bash

# 0) Setup Kafka Enviroment ==============================================================

# docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
#   --bootstrap-server localhost:9092 \
#   --create --topic iris-input --partitions 3 --if-not-exists

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
  --create --topic iris-output --partitions 1 --if-not-exists

# === Describe topic =================================================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic iris-input

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic local-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic global-weights-topic

# === Delete topic =================================================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic iris-input

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic local-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic global-weights-topic-1

docker exec broker bash -c '
  /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 --list \
  | grep "gBestStore-changelog" \
  | while read t; do
      echo "Deleting topic: $t"
      /opt/kafka/bin/kafka-topics.sh \
        --bootstrap-server localhost:9092 \
        --delete --topic "$t"
    done
'

# === Check the topics are there ====================================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list 

# === Delete Kafka Records ==========================================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic iris-input

# ===================================================================================================
######################################## Start of run ###############################################
# ===================================================================================================

# 1) =====================================================================================
docker compose up

# 2) =====================================================================================

# python3 run_pso.py

# mvn -q -DskipTests package                 
# java -jar target/iris-streams-1.0.0.jar    

mvn -q -DskipTests clean compile exec:java

# 3) =====================================================================================

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-input --from-beginning
  
docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-output --from-beginning

# 4) =====================================================================================

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic local-weights-topic --from-beginning

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic global-weights-topic --from-beginning

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic pbest-weights-topic --from-beginning


# 5) =====================================================================================

python3 iris_data_producer.py

# 6) Reset those with Application ===================================================================

docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
  --application-id pso-worker \
  --input-topics iris-input \
  --bootstrap-server localhost:9092 \
  --force




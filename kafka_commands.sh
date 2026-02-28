#! /bin/bash

# ==============================================================
# recreate:

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic pbest-weights-topic --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic local-weights-topic --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic global-weights-topic --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic prediction-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic prediction-output --partitions 1 --if-not-exists

# Verity retention policy: 
docker exec -it broker bash -lc "grep -E 'log.retention|retention.bytes' -n /etc/kafka/server.properties /opt/kafka/config/server.properties 2>/dev/null || true"

# Check that there is no in topic override
docker exec -it broker bash -lc '
/opt/kafka/bin/kafka-configs.sh --bootstrap-server localhost:9092 \
  --entity-type topics --entity-name pendigits-half-input --describe
'
# ==============================================================
# Evaluate position:

kafka-topics.sh --bootstrap-server localhost:19092 --list 

docker exec -it broker \
  /opt/kafka/bin/kafka-run-class.sh \
  org.apache.kafka.tools.GetOffsetShell \
  --bootstrap-server localhost:9092 \
  --topic susy-input \
  --time -1

docker exec -it broker \
  /opt/kafka/bin/kafka-run-class.sh \
  org.apache.kafka.tools.GetOffsetShell \
  --bootstrap-server localhost:9092 \
  --topic susy-test \
  --time -1
  
exit 0

#  ==============================================================

# docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
#   --bootstrap-server localhost:9092 \
#   --create --topic iris-input --partitions 3 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --list 

# Delete old StateStores ===================================================================

for app in pso-worker pso-coordinator; do
  docker exec -it broker /opt/kafka/bin/kafka-streams-application-reset.sh \
    --application-id "$app" \
    --input-topics iris-input \
    --bootstrap-server localhost:9092 \
    --force
done

# ========================================================================
# ========================================================================
# DATA_TOPIC =============================================================
# ========================================================================
# ========================================================================

# iris: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic iris-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic iris-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic iris-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-input --from-beginning
  
# iris-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic iris-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic iris-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic iris-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-test --from-beginning
  

# wine: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic wine-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic wine-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic wine-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic wine-input --from-beginning
  
# wine-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic wine-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic wine-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic wine-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic wine-test --from-beginning

# mnist: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic mnist-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --describe --topic mnist-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic mnist-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic mnist-input --from-beginning

# mnist-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic mnist-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic mnist-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic mnist-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic mnist-test --from-beginning

# mnist4: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic mnist4-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --describe --topic mnist4-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic mnist4-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic mnist4-input --from-beginning

# mnist4-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic mnist4-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic mnist4-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic mnist4-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic mnist4-test --from-beginning

# fashion_mnist: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic fashion_mnist-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --describe --topic fashion_mnist-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic fashion_mnist-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic fashion_mnist-input --from-beginning

# fashion_mnist-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic fashion_mnist-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic fashion_mnist-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic fashion_mnist-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic fashion_mnist-test --from-beginning

# susy: ======================================================

# docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
#   --bootstrap-server localhost:9092 \
#   --create --topic susy-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic susy-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --describe --topic susy-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic susy-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic susy-input --from-beginning

# susy test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic susy-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --describe --topic susy-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic susy-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic susy-test --from-beginning

# winequality: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic winequality-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --describe --topic winequality-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic winequality-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic winequality-input --from-beginning

# winequality test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic winequality-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --describe --topic winequality-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic winequality-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic winequality-test --from-beginning


# letter: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic letter-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic letter-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic letter-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic letter-input --from-beginning
  
# letter-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic letter-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic letter-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic letter-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic letter-test --from-beginning

# bank: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic bank-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic bank-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic bank-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic bank-input --from-beginning

# bank test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic bank-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic bank-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic bank-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic bank-test --from-beginning

# adult: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic adult-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic adult-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic adult-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic adult-input --from-beginning

# adult test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic adult-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic adult-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic adult-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic adult-test --from-beginning

# covertype: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic covertype-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic covertype-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic covertype-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic covertype-input --from-beginning

# covertype test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic covertype-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic covertype-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic covertype-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic covertype-test --from-beginning

# har: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic har-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic har-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic har-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic har-input --from-beginning
  
# har-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic har-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic har-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic har-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic har-test --from-beginning

# pendigits: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic pendigits-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic pendigits-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic pendigits-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic pendigits-input --from-beginning
  
# pendigits-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic pendigits-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic pendigits-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic pendigits-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic pendigits-test --from-beginning

# pendigits-half: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic pendigits-half-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic pendigits-half-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic pendigits-half-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic pendigits-half-input --from-beginning
  
# pendigits-half-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic pendigits-half-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic pendigits-half-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic pendigits-half-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic pendigits-half-test --from-beginning

# cifar3: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic cifar3-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic cifar3-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic cifar3-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic cifar3-input --from-beginning
  
# cifar3-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic cifar3-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic cifar3-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic cifar3-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic cifar3-test --from-beginning

# cifar5: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic cifar5-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic cifar5-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic cifar5-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic cifar5-input --from-beginning
  
# cifar5-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic cifar5-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic cifar5-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic cifar5-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic cifar5-test --from-beginning

# cifar10: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic cifar10-input --partitions 40 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic cifar10-input

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic cifar10-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic cifar10-input --from-beginning
  
# cifar10-test: ======================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic cifar10-test --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic cifar10-test

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic cifar10-test

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic cifar10-test --from-beginning

# ========================================================================
# ========================================================================
# ========================================================================
# ========================================================================

# PBEST_WEIGHTS_TOPIC ==============================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic pbest-weights-topic --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic pbest-weights-topic

kafka-topics.sh --bootstrap-server localhost:19092 \
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

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic local-weights-topic --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic local-weights-topic

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic local-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic local-weights-topic --from-beginning

# GPEST_WEIGHTS_TOPIC ==============================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic global-weights-topic --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic global-weights-topic

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic global-weights-topic

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic global-weights-topic --from-beginning

# PREDICTION_INPUT_TOPIC ==============================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic prediction-input --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic prediction-input

kafka-topics.sh --bootstrap-server localhost:19092 --describe --topic prediction-input

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction-input

# PREDICTION_OUTPUT_TOPIC ==============================================================

kafka-topics.sh --bootstrap-server localhost:19092 \
  --create --topic prediction-output --partitions 1 --if-not-exists

kafka-topics.sh --bootstrap-server localhost:19092 \
  --delete --topic prediction-output

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction-output

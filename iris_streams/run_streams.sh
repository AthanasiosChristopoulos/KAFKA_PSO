#!/bin/bash

export NUM_WORKERS=5

# --- Reset section -----------------------------------------------------------
# if [[ "$1" == "--reset" ]]; then    # execute only if there is a reset flag

    # for ((i=0; i<NUM_WORKERS; i++)); do
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

# fi

export BATCH_SIZE=150        # After BATCH_SIZE we evaluate the accuracy
export N_BATCHES=10      # After read N BATCH_SIZE, then send to Kafka

export DATA_TOPIC=iris-input
export LOCAL_WEIGHTS_TOPIC=local-weights-topic
export GLOBAL_WEIGHTS_TOPIC=global-weights-topic

export DESIRED_ACCURACY=0.87
export W_INERTIA=0.95
export C=1.7     
export C1=1.0     
export C2=2.0     

# mvn -q -DskipTests clean compile exec:java

mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java
# mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java
# mvn -q -DskipTests -Dexec.mainClass=evaluate.ExportDl4jModel clean compile exec:java

# mvn -q -DskipTests package                 
# java -jar target/iris-streams-1.0.0.jar    







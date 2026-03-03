```bash
java -version
mvn -version
mkdir -p ~/streams-demo/src/main/java/demo

mvn -q -DskipTests package
java -jar target/iris-streams-1.0.0.jar
```

When a Kafka client (like kafka-topics.sh, a producer, or consumer) wants to talk to a Kafka cluster it needs a broker addres. 
<code> --bootstrap-server localhost:9092 \ </code>
That first contact is called the bootstrap server. Connect to the Kafka broker that’s listening at localhost:9092.

```bash
docker version

docker ps       # ps = process status, print all running containers (with their status)
                # docker ps -a , means show "all" containers even stopped ones 

docker compose up     
	# If containers do NOT exist:
		# Docker creates new containers from scratch (from the docker-compose.yml	).
	# If containers already exist => Docker reuses them.
	
docker compose up -d    # -d => means detached mode 
docker compose down   		  
	# Docker will:
		# Stop all containers in that compose project
		# Delete those containers
		# Delete the default network

# reset / stop → delete → recreate:
docker compose down -v # delete the volumes as well
rm -rf ./data
docker compose up

# restart (dont delete the containers):
docker compose restart

# these run stuff kafka from inside docker (docker sends relays the commands to kafka)

# docker exec -it broker => open a shell inside the container named "broker" and cd to "kafka-topics.sh"
# kafka-topics.sh is the CLI tool for creating kafka topics
docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic iris-input --partitions 1 --replication-factor 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic iris-output --partitions 1 --replication-factor 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic iris-output --partitions 1 --replication-factor 1 --if-not-exists

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list  # this just prints the kafka topics 

# ===================================================================================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --describe --topic iris-input     # tells you metadata about the topic iris-input.
        # Leader: 1 => this tells you the Broker ID that currently leads (the one that actually writes and reads) that partition 
        # Replicas: 1 => these are the broker IDs that keep a copy of that partition. In this case only 1 Broker has that partition
        # PartitionCount: 1 => amount of partitions this topic is made out of
        # ReplicationFactor: 1 => there is 1 total copies of the data (since there is only 1 Broker)
        # Partition: 0 => This is the partition number (partition ID), each topic starts counting partitions from 0

# ===================================================================================================================

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-input --from-beginning  # This opens a consumer that prints whatever is and is received in the topic iris-input from the begging 
    # by default this displays only values no keys. Kafka Record Key a small piece of metadata Kafka uses for partitioning (Partitioning (same key → same partition)) and joins (optional, you can just do key == null in a simple event stream without multiple partitions).

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-output --from-beginning

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-input    # Start reading only new messages that arrive after I open the consumer

docker exec -it broker /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic iris-output

# ===================================================================================================================

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic iris-input

docker exec -it broker /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic iris-output

# ===================================================================================================================
# Volume stuff - Where is Kafka Logging Writing:
docker volume ls  # Find all docker volumes
docker volume inspect kafka_pso_4_kafka_data  # Inspect volume size => too small kafka is writing somewhere else

# Find Kafka Log Dirs:
docker exec -it broker bash -lc 'echo "LOG_DIRS="; grep -E "^\s*log\.dirs\s*=" -n /opt/kafka/config/server.properties /etc/kafka/server.properties 2>/dev/null || true; echo; ps aux | grep -E "kafka\.Kafka|log\.dirs" | grep -v grep || true'

# We see that docker writes inside /tmp/kafka-logs, which is inside the container not the host. Docker keeps the container filesystem, doesnt delete it.
docker exec -it broker sh -lc       # runs it as a shell inside the docker container   
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs'
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs/*'  # show per partition

# Volumes / bind mounts
# These are stored outside the container lifecycle and are meant for persistence (independent from what the container does).
  # - Volumes => data survives container deletion and recreation
  # Without a volume, data survives a restart, but not a container shutdown with docker compose down, docker rm broker
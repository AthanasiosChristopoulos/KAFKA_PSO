# PSO-Streams — Distributed PSO Training of Neural Networks via Kafka Streams

> **Thesis Project** — Particle Swarm Optimization (PSO) for neural network training on classification datasets, built on Apache Kafka, Kafka Streams, and DL4J.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Run with Docker](#run-with-docker)
  - [Run Manually](#run-manually)
- [Running the Project](#running-the-project)
  - [Normal Training](#normal-training)
  - [Experimentation Mode](#experimentation-mode)
  - [Build & Execute JAR](#build--execute-jar)
- [Kafka Setup](#kafka-setup)
  - [Docker (Recommended)](#docker-recommended)
  - [Manual Kafka Setup (Linux)](#manual-kafka-setup-linux)
  - [Kafka for Windows](#kafka-for-windows)
  - [Federated / Remote Execution](#federated--remote-execution)
- [Python Dependencies](#python-dependencies)
- [Utilities](#utilities)
  - [GPU Monitoring](#gpu-monitoring)
  - [Docker Storage Inspection](#docker-storage-inspection)
  - [Keras 2 (.h5) Files on Ubuntu](#keras-2-h5-files-on-ubuntu)
  - [Network Latency Estimation](#network-latency-estimation)
  - [Clock Synchronization](#clock-synchronization)

---

## Overview

This project implements **PSO-based training of neural networks** for dataset classification tasks. It is designed to:

- Train on UCI and other standard multi-feature classification datasets
- Support Feedforward Neural Networks (FNN) and Convolutional Neural Networks (CNN)
- Enable the use of **non-differentiable fitness/loss functions** (a key advantage of PSO over gradient-based methods)
- Distribute the workload across multiple Kafka Streams workers for parallel training

Two PSO variants are supported:

- **Classical PSO** (best-of-neighbourhood)
- **Fully Informed PSO (FIPS)**

---

## Architecture

The system is built on top of **Apache Kafka**, using **Kafka Streams** workers (Java/DL4J) and Python producers/consumers.

### Kafka Topics

| Topic | Purpose |
|---|---|
| `DATA_TOPIC` | Partitioned training dataset |
| `TEST_TOPIC` | Test/evaluation data |
| `PBEST_WEIGHTS_TOPIC` | Personal best particle weights |
| `LOCAL_WEIGHTS_TOPIC` | Local (per-worker) model weights |
| `GBEST_WEIGHTS_TOPIC` | Global best particle weights |
| `PREDICTION_INPUT_TOPIC` | Inference input |
| `PREDICTION_OUTPUT_TOPIC` | Inference output |

### How It Works

- The training dataset is partitioned across `DATA_TOPIC`
- Each **Kafka Streams worker** reads from its assigned partition(s), training on its own local model
- Workers participate in PSO by exchanging weight vectors via the weight topics
- This partitioning scheme enables **data parallelism** across workers

```
┌──────────────────────────────────────────────────────────┐
│                        Kafka Broker                      │
│  DATA_TOPIC (partitioned)  │  PBEST / LOCAL / GBEST      │
└────────────┬───────────────┴──────────────┬──────────────┘
             │                              │
     ┌───────▼────────┐            ┌────────▼───────┐
     │  Worker 0      │            │  Worker N      │
     │  (Partition 0) │   <PSO>    │  (Partition N) │
     │  DL4J Model    │ ◄────────► │  DL4J Model    │
     └────────────────┘            └────────────────┘
```

---

## Tech Stack

- **Java** (Maven) — Kafka Streams workers, DL4J model training
- **Apache Kafka / Kafka Streams** — distributed messaging and stream processing
- **DL4J (Deeplearning4j)** — neural network inference and weight management
- **Python** — data producers, consumers, preprocessing, Keras model conversion
- **Docker** — Kafka broker containerization

---

## Getting Started

### Prerequisites

- Java 11+, Maven
- Docker & Docker Compose
- Python 3.x

### Run with Docker

```bash
# Start Kafka broker
docker compose up

# Stop
docker compose stop
```

### Run Manually

Ensure Kafka is running (see [Kafka Setup](#kafka-setup)), then proceed to [Running the Project](#running-the-project).

---

## Running the Project

### Normal Training

Reads configuration from `.env`:

```bash
./run_streams.sh
```

Or manually:

```bash
mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java
```

### Experimentation Mode

Runs the experiment selected in `.env`:

```bash
./run_streams_exp.sh
```

Or manually:

```bash
mvn -q -DskipTests -Dexec.mainClass=experimentation.Experimentation clean compile exec:java
```

### Build & Execute JAR

**Package:**

```bash
mvn -DskipTests package
```

**Run (Linux):**

```bash
# Run main class
java -jar target/pso-streams-1.0.0.jar

# Run a specific worker (e.g. worker 0)
java -cp target/pso-streams-1.0.0.jar pso.WorkerMain 0

# With memory limits and diagnostics
/usr/bin/time -v java \
  -Xms256m -Xmx768m \
  -XX:+UnlockDiagnosticVMOptions \
  -XX:NativeMemoryTracking=summary \
  -cp target/pso-streams-1.0.0.jar \
  pso.WorkerMain 0

# Suppress verbose logging
java \
  -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
  -cp target/pso-streams-1.0.0.jar \
  pso.WorkerMain 0 1 2 3

# Run 12 workers
java \
  -Dorg.slf4j.simpleLogger.defaultLogLevel=error \
  -cp target/pso-streams-1.0.0.jar \
  pso.WorkerMain 0 1 2 3 4 5 6 7 8 9 10 11
```

**Run (Windows):**

```bat
java ^
  -Dorg.slf4j.simpleLogger.defaultLogLevel=error ^
  -cp target\pso-streams-1.0.0.jar ^
  pso.WorkerMain 0 1 2 3

REM Compile and run directly (Windows):
mvn -e -DskipTests ^
  -Dexec.mainClass=pso.WorkerMain ^
  -Dexec.args="0 1 2 3" ^
  -Dorg.slf4j.simpleLogger.defaultLogLevel=error ^
  -Pcpu ^
  compile exec:java
```

---

## Kafka Setup

### Docker (Recommended)

A `docker-compose.yml` is provided:

```bash
docker compose up
docker compose stop
```

### Manual Kafka Setup (Linux)

**Download and install:**

```bash
mkdir -p ~/tools && cd ~/tools
wget https://archive.apache.org/dist/kafka/3.7.0/kafka_2.13-3.7.0.tgz
tar -xzf kafka_2.13-3.7.0.tgz
mv kafka_2.13-3.7.0 kafka-local

export KAFKA_HOME=~/tools/kafka-local
export PATH="$KAFKA_HOME/bin:$PATH"

# Make permanent (optional)
echo 'export KAFKA_HOME="$HOME/tools/kafka-local"' >> ~/.bashrc
echo 'export PATH="$KAFKA_HOME/bin:$PATH"' >> ~/.bashrc
```

**Configure KRaft mode** (`~/tools/kafka-local/config/kraft/server.properties`):

```properties
log.dirs=/home/<your-user>/kafka-kraft/logs
log.retention.hours=-1
log.retention.bytes=2221225472
group.initial.rebalance.delay.ms=500
```

**Format and start:**

```bash
# Generate a cluster ID
bin/kafka-storage.sh random-uuid

# Format storage
bin/kafka-storage.sh format \
  --cluster-id <YOUR_CLUSTER_ID> \
  --config ~/tools/kafka-local/config/kraft/server.properties

# Start the broker
kafka-server-start.sh ~/tools/kafka-local/config/kraft/server.properties

# Reduce log noise (optional)
sed -i 's/=INFO/=WARN/g' ~/tools/kafka-local/config/log4j.properties
```

**Switch between localhost and network IP:**

```bash
# Localhost → network IP
sed -i 's/127\.0\.0\.1/YOUR_IP/g' server.properties
sed -i 's/localhost/YOUR_IP/g' server.properties

# Network IP → localhost
sed -i 's/YOUR_IP/localhost/g' server.properties
sed -i 's/YOUR_IP/127.0.0.1/g' server.properties
```

**Test connectivity:**

```bash
kafka-topics.sh --bootstrap-server YOUR_IP:9092 --list

kafka-console-consumer.sh --bootstrap-server YOUR_IP:19092 \
  --topic pendigits-input --from-beginning
```

### Kafka for Windows

```bat
C:\tools\kafka-local\bin\windows\kafka-topics.bat --bootstrap-server YOUR_IP:19092 --list

C:\tools\kafka-local\bin\windows\kafka-console-consumer.bat ^
  --bootstrap-server YOUR_IP:19092 ^
  --topic pendigits-input ^
  --from-beginning
```

### Federated / Remote Execution

Find your LAN IP:

```bash
hostname -I
# or
ip addr
```

Set in `server.properties`:

```properties
controller.quorum.voters=1@<lan_ip>:9093
listeners=PLAINTEXT://:9092,CONTROLLER://:9093
advertised.listeners=PLAINTEXT://<lan_ip>:9092
```

---

## Python Dependencies

```bash
python3 -m pip install \
  tensorflow \
  scikit-learn \
  kafka-python \
  numpy \
  pandas \
  pillow \
  scipy \
  tensorflow-datasets
```

---

## Utilities

### GPU Monitoring

```bash
nvidia-smi
watch -n 0.5 nvidia-smi
watch -n 1 -t nvidia-smi \
  --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv
```

### Docker Storage Inspection

```bash
# Open a shell inside the Kafka broker container
docker exec -it broker sh -lc

# Check Kafka log size
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs'
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs/*'
```

### Keras 2 (.h5) Files on Ubuntu

Keras 3 breaks `.h5` compatibility. Use a Python 3.11 venv with TF 2.15:

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

source ~/venvs/tf215/bin/activate
```

### Network Latency Estimation

```bash
# Localhost
ping -c 30 -s 4060 127.0.0.1
# Expected: rtt avg ~0.027 ms

# LAN
ping -c 30 -s 4060 192.168.2.15
# Expected: rtt avg ~70 ms (varies with load)
```

### Clock Synchronization

**Windows:**

```bat
w32tm /stripchart /computer:192.168.2.15 /samples:5 /dataonly
w32tm /config /manualpeerlist:"192.168.2.15,0x8" /syncfromflags:manual /update
net stop w32time && net start w32time
w32tm /resync
w32tm /query /status
```

**Ubuntu:**

```bash
chronyc clients
chronyc tracking
chronyc sources
```

---

## Maven Tips

```bash
# Check ND4J dependency tree
mvn dependency:tree | grep nd4j
```
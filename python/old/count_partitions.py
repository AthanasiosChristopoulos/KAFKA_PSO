from kafka import KafkaConsumer, TopicPartition
import os

# ========================================================================================
# Env + Args =============================================================================

from dotenv import load_dotenv
loaded = load_dotenv("../java/.env")
print("Dotenv loaded:", loaded)

DATASET = os.getenv("DATASET")

BATCH_FLUSH = int(os.getenv("BATCH_FLUSH"))

INPUT_TOPIC = DATASET + "-input"
TEST_TOPIC = DATASET + "-test"

print(f"Running this on input topic: {INPUT_TOPIC}")

# ========================================================================================
# Kafka Producer =========================================================================

BOOTSTRAP = "localhost:9092"

def main():
    c = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP,
        enable_auto_commit=False,
        group_id=None,              # no consumer group needed
        api_version_auto_timeout_ms=30000,
    )

    partitions = sorted(list(c.partitions_for_topic(INPUT_TOPIC) or []))
    if not partitions:
        raise RuntimeError(f"Topic not found or no partitions: {INPUT_TOPIC}")

    tps = [TopicPartition(INPUT_TOPIC, p) for p in partitions]
    c.assign(tps)

    begins = c.beginning_offsets(tps)
    ends   = c.end_offsets(tps)

    total = 0
    print(f"Topic: {INPUT_TOPIC}")
    for tp in tps:
        b = begins.get(tp, 0)
        e = ends.get(tp, 0)
        n = e - b
        total += n
        print(f"  partition {tp.partition:02d}: begin={b} end={e} count={n}")

    print(f"TOTAL records across all partitions: {total}")
    c.close()

if __name__ == "__main__":
    main()

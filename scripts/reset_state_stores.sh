#!/bin/bash
set -euo pipefail

BROKER="broker"
BOOTSTRAP="localhost:19092"

# Match either prefix
PREFIX_REGEX="(pso-coordinator|pso-gbest)"
SUFFIX_REGEX="(-gBestEmitStore-changelog|-test-data-store-changelog)"

echo "Searching for topics matching:"
echo "  ${PREFIX_REGEX}*${SUFFIX_REGEX}"
echo

TOPICS=$(kafka-topics.sh --bootstrap-server "$BOOTSTRAP" --list | grep -E "^${PREFIX_REGEX}.*${SUFFIX_REGEX}$" || true)

if [[ -z "$TOPICS" ]]; then
  echo "No matching topics found."
  exit 0
fi

echo "Found the following topics:"
echo "$TOPICS"
echo

for topic in $TOPICS; do
  echo "Deleting: $topic"
  kafka-topics.sh \
    --bootstrap-server "$BOOTSTRAP" \
    --delete \
    --topic "$topic" || true
done

echo
echo "Done deleting changelog topics."

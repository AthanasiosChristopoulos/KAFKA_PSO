package utils;

import org.apache.kafka.clients.admin.*;
import org.apache.kafka.common.errors.TopicExistsException;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ExecutionException;

public final class KafkaTopicManager {

    private KafkaTopicManager() {}

    //============================================================================

    public static AdminClient admin(String bootstrapServers) {
        Properties props = new Properties();
        props.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(AdminClientConfig.REQUEST_TIMEOUT_MS_CONFIG, "15000");
        props.put(AdminClientConfig.DEFAULT_API_TIMEOUT_MS_CONFIG, "20000");
        return AdminClient.create(props);
    }

    //===================================================================================

    public static void recreateTopics(String bootstrapServers, List<String> topics, int partitions, int replicationFactor) throws Exception {
        try (AdminClient admin = admin(bootstrapServers)) {
            deleteTopics(admin, topics);
            createTopics(admin, topics, partitions, replicationFactor);
        }
    }

    //===================================================================================

    public static void deleteTopics(AdminClient admin, List<String> topics) throws Exception {
        Set<String> existing = admin.listTopics().names().get();
        List<String> toDelete = topics.stream().filter(existing::contains).toList();
        if (toDelete.isEmpty()) return;

        admin.deleteTopics(toDelete).all().get();
        waitUntilDeleted(admin, new HashSet<>(toDelete), Duration.ofSeconds(20));
    }

    //=====================================================================

    private static void waitUntilDeleted(AdminClient admin, Set<String> topics, Duration maxWait) throws Exception {

        long deadline = System.nanoTime() + maxWait.toNanos();
        while (System.nanoTime() < deadline) {
            Set<String> existing = admin.listTopics().names().get();
            boolean anyStillThere = topics.stream().anyMatch(existing::contains);
            if (!anyStillThere) return;
            Thread.sleep(200);
        }
        throw new RuntimeException("Timed out waiting for topics to delete: " + topics);
    }

    //===================================================================================================
    
    public static void createTopics(AdminClient admin, List<String> topics, int partitions, int replicationFactor) throws Exception {

        List<NewTopic> newTopics = new ArrayList<>();
        for (String t : topics) {
            newTopics.add(new NewTopic(t, partitions, (short) replicationFactor));
        }

        try {
            admin.createTopics(newTopics).all().get();

        } catch (ExecutionException e) {
            if (e.getCause() instanceof TopicExistsException) {
                return;
            }
            throw e;
        }
    }
}

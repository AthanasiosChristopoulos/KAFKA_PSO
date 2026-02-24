// package utils;

// import org.apache.kafka.clients.admin.*;
// import org.apache.kafka.common.errors.TopicExistsException;

// import java.time.Duration;
// import java.util.*;
// import java.util.concurrent.ExecutionException;

// public final class KafkaTopicManager {

//     private KafkaTopicManager() {}

//     public static AdminClient admin(String bootstrapServers) {
//         Properties props = new Properties();
//         props.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
//         // Optional timeouts:
//         props.put(AdminClientConfig.REQUEST_TIMEOUT_MS_CONFIG, "15000");
//         props.put(AdminClientConfig.DEFAULT_API_TIMEOUT_MS_CONFIG, "20000");
//         return AdminClient.create(props);
//     }

//     public static void recreateTopics(
//             String bootstrapServers,
//             List<String> topics,
//             int partitions,
//             short replicationFactor
//     ) throws Exception {
//         try (AdminClient admin = admin(bootstrapServers)) {
//             deleteTopics(admin, topics);
//             createTopics(admin, topics, partitions, replicationFactor);
//         }
//     }

//     public static void deleteTopics(AdminClient admin, List<String> topics) throws Exception {
//         // Delete only topics that exist (to avoid noisy errors)
//         Set<String> existing = admin.listTopics().names().get();
//         List<String> toDelete = topics.stream().filter(existing::contains).toList();
//         if (toDelete.isEmpty()) return;

//         admin.deleteTopics(toDelete).all().get();

//         // Wait until deletion is actually visible (Kafka deletes async)
//         waitUntilDeleted(admin, new HashSet<>(toDelete), Duration.ofSeconds(20));
//     }

//     private static void waitUntilDeleted(AdminClient admin, Set<String> topics, Duration maxWait) throws Exception {
//         long deadline = System.nanoTime() + maxWait.toNanos();
//         while (System.nanoTime() < deadline) {
//             Set<String> existing = admin.listTopics().names().get();
//             boolean anyStillThere = topics.stream().anyMatch(existing::contains);
//             if (!anyStillThere) return;
//             Thread.sleep(200);
//         }
//         // If it times out, you can still attempt create with --if-not-exists behavior,
//         // but it's better to fail fast so experiments are consistent.
//         throw new RuntimeException("Timed out waiting for topics to delete: " + topics);
//     }

//     public static void createTopics(
//             AdminClient admin,
//             List<String> topics,
//             int partitions,
//             short replicationFactor
//     ) throws Exception {
//         List<NewTopic> newTopics = new ArrayList<>();
//         for (String t : topics) {
//             newTopics.add(new NewTopic(t, partitions, replicationFactor));
//         }

//         try {
//             admin.createTopics(newTopics).all().get();
//         } catch (ExecutionException e) {
//             if (e.getCause() instanceof TopicExistsException) {
//                 // fine, behave like --if-not-exists
//                 return;
//             }
//             throw e;
//         }
//     }
// }

package utils;

import org.apache.kafka.clients.admin.*;
import org.apache.kafka.common.errors.TopicExistsException;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ExecutionException;

public final class KafkaTopicManager {

    private KafkaTopicManager() {}

    public static AdminClient admin(String bootstrapServers) {
        Properties props = new Properties();
        props.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(AdminClientConfig.REQUEST_TIMEOUT_MS_CONFIG, "15000");
        props.put(AdminClientConfig.DEFAULT_API_TIMEOUT_MS_CONFIG, "20000");
        return AdminClient.create(props);
    }

    /** Recreate topics: delete (confirmed) then create (confirmed). */
    public static void recreateTopics(
            String bootstrapServers,
            List<String> topics,
            int partitions,
            short replicationFactor,
            Duration deleteTimeout,
            Duration createTimeout
    ) throws Exception {

        Objects.requireNonNull(topics, "topics");
        if (topics.isEmpty()) return;

        try (AdminClient admin = admin(bootstrapServers)) {
            System.out.println("[KafkaTopicManager] Existing topics (before): " + existingTopics(admin));

            DeleteReport del = deleteTopicsConfirmed(admin, topics, deleteTimeout);
            System.out.println("[KafkaTopicManager] Delete report: " + del);

            CreateReport cre = createTopicsConfirmed(admin, topics, partitions, replicationFactor, createTimeout);
            System.out.println("[KafkaTopicManager] Create report: " + cre);

            System.out.println("[KafkaTopicManager] Existing topics (after): " + existingTopics(admin));
        }
    }

    /** Returns a set of existing topic names. */
    public static Set<String> existingTopics(AdminClient admin) throws Exception {
        return admin.listTopics(new ListTopicsOptions().listInternal(true)).names().get();
    }

    /** Delete topics and WAIT until they are actually gone (or timeout). */
    public static DeleteReport deleteTopicsConfirmed(
            AdminClient admin,
            List<String> topics,
            Duration timeout
    ) throws Exception {

        long deadlineNanos = System.nanoTime() + timeout.toNanos();

        Set<String> existing = existingTopics(admin);
        List<String> toDelete = topics.stream().filter(existing::contains).toList();

        if (toDelete.isEmpty()) {
            return new DeleteReport(topics, List.of(), true, "Nothing to delete (topics not found).");
        }

        // Request delete
        admin.deleteTopics(toDelete).all().get();

        // Confirm delete (async): poll until they disappear
        while (System.nanoTime() < deadlineNanos) {
            Set<String> now = existingTopics(admin);
            boolean anyStillThere = toDelete.stream().anyMatch(now::contains);
            if (!anyStillThere) {
                return new DeleteReport(topics, toDelete, true, "Deleted and confirmed.");
            }
            Thread.sleep(250);
        }

        // Still exists -> fail with details
        Set<String> still = existingTopics(admin);
        List<String> stillThere = toDelete.stream().filter(still::contains).toList();
        return new DeleteReport(topics, stillThere, false,
                "Timed out waiting for deletion. Still present: " + stillThere);
    }

    /** Create topics and confirm they exist AND have the desired partition count (or timeout). */
    public static CreateReport createTopicsConfirmed(
            AdminClient admin,
            List<String> topics,
            int partitions,
            short replicationFactor,
            Duration timeout
    ) throws Exception {

        long deadlineNanos = System.nanoTime() + timeout.toNanos();

        // Create (behave like --if-not-exists)
        List<NewTopic> newTopics = topics.stream()
                .map(t -> new NewTopic(t, partitions, replicationFactor))
                .toList();

        try {
            admin.createTopics(newTopics).all().get();
        } catch (ExecutionException e) {
            if (!(e.getCause() instanceof TopicExistsException)) {
                throw e;
            }
            // Topic exists is okay: we'll verify anyway
        }

        // Confirm create: poll describeTopics + verify partitions
        while (System.nanoTime() < deadlineNanos) {

            try {
                Map<String, TopicDescription> desc = admin.describeTopics(topics).allTopicNames().get();

                // Verify all topics exist and have expected partition count
                List<String> bad = new ArrayList<>();
                for (String t : topics) {
                    TopicDescription td = desc.get(t);
                    if (td == null) {
                        bad.add(t + "(missing)");
                        continue;
                    }
                    int actualPartitions = td.partitions().size();
                    if (actualPartitions != partitions) {
                        bad.add(t + "(partitions=" + actualPartitions + ")");
                    }
                }

                if (bad.isEmpty()) {
                    return new CreateReport(topics, true, "Created and confirmed partitions=" + partitions);
                }

            } catch (ExecutionException ex) {
                // Happens if a topic isn't visible yet
                // Keep polling until timeout
            }

            Thread.sleep(250);
        }

        // Final status on timeout
        Set<String> now = existingTopics(admin);
        List<String> missing = topics.stream().filter(t -> !now.contains(t)).toList();
        return new CreateReport(topics, false,
                "Timed out waiting for creation. Missing: " + missing);
    }
    
    public static DeleteReport deleteTopicsConfirmed(
            String bootstrapServers,
            List<String> topics,
            Duration timeout
    ) throws Exception {
        Objects.requireNonNull(topics, "topics");
        if (topics.isEmpty()) {
            return new DeleteReport(List.of(), List.of(), true, "No topics requested.");
        }
        try (AdminClient admin = admin(bootstrapServers)) {
            DeleteReport del = deleteTopicsConfirmed(admin, topics, timeout);
            System.out.println("[KafkaTopicManager] Delete report: " + del);
            return del;
        }
    }

    // Convenience: delete ONE topic
    public static DeleteReport deleteTopicConfirmed(
            String bootstrapServers,
            String topic,
            Duration timeout
    ) throws Exception {
        return deleteTopicsConfirmed(bootstrapServers, List.of(topic), timeout);
    }

    // ------------------------------------------------------------------------------------
    // Simple report objects (records) for logging and debugging
    // ------------------------------------------------------------------------------------

    public record DeleteReport(
            List<String> requested,
            List<String> affectedOrStillThere,
            boolean success,
            String message
    ) {}

    public record CreateReport(
            List<String> requested,
            boolean success,
            String message
    ) {}
}
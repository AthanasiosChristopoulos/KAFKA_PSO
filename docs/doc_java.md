## Two different tasks: compile vs package with shade
 - For dev, you only need compile.
 - For distribution, you need shade (IoT devices dont compile, they execute compiled code) => builds a JAR.

✅ Use mvn package only when you want a new JAR (or after changing dependencies).
✅ Use mvn exec:java for day-to-day runs of the code.

**mvn -q -DskipTests package && java -jar target/iris-streams-1.0.0.jar**
    Maven runs the whole lifecycle up to package, including the maven-shade-plugin you configured.

    Shade does this (Heavy):
        Takes your compiled classes
        Unpacks all dependencies (Kafka, DL4J, ND4J native, Jackson, etc.)
        Repacks them into one big fat JAR

    The Java compile step itself is relatively fast.
    The shading is what’s taking ages.

**mvn -q -DskipTests clean compile exec:java**
    - Used for Development (no distribution of this code to the IoT devices)
    - Clean => deletes whatever (the artifacts / jars) is on the target/ (directory of maven)
    - compile (create compiled .class, put them in target/classes)
    - exec:java (run the code)

**Artifacts:**
 
 - An artifact is any file produced or used by Maven.
    - It has the attributes: groupId + artifactId + version + packaging 
 - In a maven project they are stored under target/ directory
 - .jar is a type of artifact (it has the .jar packaging)
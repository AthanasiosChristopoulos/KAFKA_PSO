
# Java Project - Maven ==========================================

## Two different tasks: compile vs package with shade: ===========================================

 - For dev, you only need compile.
     - Use mvn exec:java for day-to-day runs of the code.
 - For distribution, you need shade (IoT devices dont compile, they execute compiled code from a JAR) => shading builds a JAR.
     - Use mvn package only when you want a new JAR (or after changing dependencies).

```bash
mvn -q -DskipTests package && java -jar target/iris-streams-1.0.0.jar :
```    
    Maven runs the whole lifecycle up to package, including the maven-shade-plugin you configured.

    Shade does this (Heavy):
        Takes your compiled classes
        Unpacks all dependencies (Kafka, DL4J, ND4J native, Jackson, etc.)
        Repacks them into one big fat JAR

    The Java compile step itself is relatively fast.
    The shading is what’s taking ages.

```bash
mvn -q -DskipTests clean compile exec:java
``` 
    - Used for Development (no distribution of this code to the IoT devices)
    - Clean => deletes the entire target directory: whatever (the artifacts / jars) is on the target/ (directory of maven)  
        - it does not force Maven to re-download dependencies
        - Without clean, Maven will recompile only if something has changed otherwise reuse target / classes
            - recompile if a .java source file changed
            - recompile if the classpath changed => dependency JARs, versions (basically the pom.xml)
    - compile (create compiled .class, put them in target/classes)
    - exec:java (run the code)

## Artifacts: ============================================================================
 
 - An artifact is any file produced or used by Maven.
    - It has the attributes: groupId + artifactId + version + packaging 
 - In a maven project they are stored under target/ directory
 - .jar is a type of artifact (it has the .jar packaging)

## pom.xml: =================================================================

Has dependency JARs with versions, artifact IDs, ... 
This is one dependency JAR: 
<dependency>
  <groupId>org.nd4j</groupId>
  <artifactId>nd4j-native-platform</artifactId>
  <version>1.0.0-M2.1</version>
</dependency>

 - The paerticular one is a platform / aggregator dependency, that brings in from the internet other dependency JARs automatically.
 - Maven will read pom.xml and build a compile classpath. Using that classpath, compile .java source files
 - Dependencies are not recompiled when you build your project. They’re already compiled JARs.

## Run with CPU or GPU: =====================================================================

Test:
 - htop => CPU / RAM only 
 - nvidia-smi

Use CPU instrution: =========================================================================
<dependency>
  <groupId>org.nd4j</groupId>
  <artifactId>nd4j-native-platform</artifactId>
  <version>1.0.0-M2.1</version>
</dependency>

Use GPU instruction: =========================================================================
This is means use CUDA instead which is:
    - CUDA = Compute Unified Device Architecture
    - It’s NVIDIA’s platform that lets software run code on the GPU instead of the CPU.
    - ND4J CUDA backend: High-level math library that internally uses CUDA

<dependency>
  <groupId>org.nd4j</groupId>
  <artifactId>nd4j-cuda-11.6-platform</artifactId>
  <version>1.0.0-M2.1</version>
</dependency>

The above gives just ND4J CUDA backend, so the code of ND4J that can use CUDA. But you need to have already installed CUDA in your machine.
Otherwise we can do this:
<dependency>
    <groupId>org.bytedeco</groupId>
    <artifactId>cuda-platform-redist</artifactId>
    <version>11.6-8.3-1.5.7</version>
</dependency>

This bundles “redistributable” CUDA libs via Maven, so it can supply the exact CUDA version that can be  used by DL4J.
It bundles CUDA native libraries (via JavaCPP/Bytedeco) so your Java app can run even if the system doesn’t have CUDA toolkit installed.
```bash
sudo apt install nvidia-cuda-toolkit
ldconfig -p | grep libcudart
```

## Memory Allocation ========================================================

Memory Allocation is expensive even when you have plenty of memory. It’s not "running out of RAM", it’s that allocating + cleaning up burns CPU, also makes GC (Garbage Collector) work harder.
 - Allocation cost by itself time, not just memory
 - GC also adds overhead and pauses 

```java
// Example: 
float[] flatProps = function(data)
// float[] flatProps => this is the reference / pointer variable, it does NOT allocate the array itself
    // this is still an allocation, but just a pointer / stack allocation, it doesnt cost anything
// function(data) is a function that might allocate memory or not and return to flatProps its starting address
    // depending on the functions internals, this might overwrite or allocate new memory

probs.getRow(i).toFloatVector();    // this function allocates new memory
probs.data().asFloat();             // this one overwrites memory

```


## Misc ===============================================================
1)
nanoTime() is guaranteed to:

✔ always increase
✔ never go backwards
✔ not affected by system clock adjustments
It only has meaning relative to another nanoTime() call.
long elapsed = t1 - t0;
This makes it ideal for performance measurement.

System.nanoTime => 324234234234234 no meaning number

2)

AtomicLong is a thread-safe long counter from java.util.concurrent.atomic.
It lets multiple threads update the same number safely without using synchronized or locks.

Normally: 
counter++ is actually three operations:
    read value
    add 1
    write value
Exampl how it could go wrong: 
    Thread A reads 5
    Thread B reads 5
    Thread A writes 6
    Thread B writes 6

Instead, AtomicLong performs the operation atomically (in one CPU instruction)

AtomicLong counter = new AtomicLong(0);
counter.incrementAndGet();  // Now multiple threads can safely update it.
long value = totalComparisons.get();    // Just reading the value
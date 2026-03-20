## ======================================================================
## Server Stuff =========================================================

```bash

ssh achristopoulos@polytechnix.softnet.tuc.gr
Erwd!oS21
# enter with:
# username: achristopoulos
# password: Erwd!oS21

ssh-keygen -t ed25519   # press enter for the options
ssh-copy-id achristopoulos@polytechnix.softnet.tuc.gr   # no need for passwords afterwards

# Launches you into:
achristopoulos@polytechnix:~$  === achristopoulos@polytechnix:/home/achristopoulos$
    => This is the home directory: ~ is the home directory of the current user

# Check also with 
pwd

# Dont use it, it will destroy kafka, because kafka is a fragile little bitch
# pkill -9 -u achristopoulos -f java 
# -9 → Force kill (SIGKILL)
# -f → Match full command line

pkill -u achristopoulos -f java   # is gracefull
pgrep -u achristopoulos -f java -a # find maven process
kill -9 <pid> # of the maven process

# Run instead: 
ps -ef | grep -i kafka | grep -v grep
kill -TERM <kafka_pid>
# Or just Ctrl + C

```

NAS = Network Attached Storage
 - High Disk Volume for every but available only through the network

```bash
cd /mnt/nas_drive/achristopoulos
cd /mnt/nas_drive/achristopoulos/projects/KAFKA_PSO_4
df -h .     # Disk Storage overall
du -sh .    # how much space your current folder uses
du -sh ~/.m2    # Directory and what space it takes
du -sh * .[!.]* | sort -rh  # Show all of them ... hidden and normal files
du -sh * | sort -rh     # all of them individually
du -sh .m2 m2repo

docker --version

git clone --branch DL4J-PSO-Generic --single-branch https://github.com/AthanasiosChristopoulos/KAFKA_PSO_4.git

# Kafka ===========================================================================

CLUSTER_ID=$(kafka-storage.sh random-uuid)
kafka-storage.sh format -t "$CLUSTER_ID" -c /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server.properties

export KAFKA_HOME=/mnt/nas_drive/achristopoulos/kafka-local
export PATH="$KAFKA_HOME/bin:$PATH"     # this isnt overriding PATH, this is are prepending to it (appending to the beggining of the list)
kafka-server-start.sh /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server.properties
kafka-server-start.sh /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server-heavy.properties

kafka-server-start.sh /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server-ssd.properties
kafka-server-start.sh /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server.properties 2>&1 | grep -Ei "error|warn"
# see differences between disks:
df -hT /home/achristopoulos/kafka-logs
df -hT /mnt/nas_drive/achristopoulos

which kafka-topics.sh   

## Try listening on see if it works:
kafka-topics.sh --bootstrap-server localhost:19092 --list

ss -ltnp | egrep ':19092|:19093|:9092|:9093'  # Check port being used

# where will kafka log ?
cd /mnt/nas_drive/achristopoulos/kafka-local/config/kraf
nano server.properties # look for the log.dirs= ... variable

# Set the following for this implementation:
log.dirs=/mnt/nas_drive/achristopoulos/kafka-kraft/logs
log.retention.hours=-1
log.retention.bytes=2221225472
group.initial.rebalance.delay.ms=500

kafka-storage.sh format \
  -t TZnQLupIQNKrZbEwke1-cw \
  -c /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server-ssd.properties

# Change Logger Level in log4j.properties
sed -i 's/=INFO/=ERROR/g' /mnt/nas_drive/achristopoulos/kafka-local/config/log4j.properties

# ========================================================================================

export JAVA_HOME=/mnt/nas_drive/achristopoulos/jdks/jdk-17.0.18+8
export PATH="$JAVA_HOME/bin:$PATH"
java --version
javac --version

ln -s /mnt/nas_drive/achristopoulos/.keras ~/.keras
ln -s /mnt/nas_drive/achristopoulos/.javacpp ~/.javacpp
ln -s /mnt/nas_drive/achristopoulos/.m2 ~/.m2
ls -ld ~/.javacpp ~/.keras  # you can see if they are links or not 

# Transfer files
scp my-pendigits.tra achristopoulos@polytechnix:/mnt/nas_drive/achristopoulos/KAFKA_PSO_4/data/
scp -r achristopoulos@polytechnix:/mnt/nas_drive/achristopoulos/KAFKA_PSO_4/java/experimental_results_server/mnist_accuracy C:\Users\User\Downloads
# On windows similarily do:
scp C:\Users\User\Downloads\nsfw_dataset_v1-20241203T200912Z-001.zip achristopoulos@polytechnix.softnet.tuc.gr:/mnt/nas_drive/achristopoulos/KAFKA_PSO_4/data

# Reset local-weights-topic:
kafka-topics.sh --bootstrap-server localhost:19092 --delete --topic local-weights-topic
kafka-topics.sh --bootstrap-server localhost:19092 --create --topic local-weights-topic --partitions 1 --if-not-exists

watch -n 1 -t nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
htop
htop -u achristopoulos
# F4 and then search for Java
ps -u achristopoulos | grep java
```

## Sync: =====================================================================
# Windows:
robocopy C:\AAAProjects\KAFKA_PSO_4\java\exp_dataset C:\AAAProjects\diplomatiki\figures\exp_dataset /MIR
robocopy C:\AAAProjects\KAFKA_PSO_4\java\exp_dataset_heavy C:\AAAProjects\diplomatiki\figures\exp_dataset_heavy /MIR

robocopy C:\AAAProjects\KAFKA_PSO_4\java\exp_dataset C:\AAAProjects\diplomatiki\figures\exp_dataset /MIR & robocopy C:\AAAProjects\KAFKA_PSO_4\java\exp_dataset_heavy C:\AAAProjects\diplomatiki\figures\exp_dataset_heavy /MIR

# Ubuntu:
rsync -av --delete /home/ds123f15/Documents/diplomatiki/KAFKA_PSO_4/java/exp_dataset/ /home/ds123f15/Documents/diplomatiki/diplomatiki_latex/diplomatiki/figures/exp_dataset/
rsync -av --delete /home/ds123f15/Documents/diplomatiki/KAFKA_PSO_4/java/exp_dataset_heavy/ /home/ds123f15/Documents/diplomatiki/diplomatiki_latex/diplomatiki/figures/exp_dataset_heavy/

rsync -av --delete /home/ds123f15/Documents/diplomatiki/KAFKA_PSO_4/java/exp_dataset/ /home/ds123f15/Documents/diplomatiki/diplomatiki_latex/diplomatiki/figures/exp_dataset/ && rsync -av --delete /home/ds123f15/Documents/diplomatiki/KAFKA_PSO_4/java/exp_dataset_heavy/ /home/ds123f15/Documents/diplomatiki/diplomatiki_latex/diplomatiki/figures/exp_dataset_heavy/

## Restore KRaft: =====================================================

```bash
rm -rf /mnt/nas_drive/achristopoulos/kafka-kraft/logs/*
cd /mnt/nas_drive/achristopoulos/kafka-local
bin/kafka-storage.sh random-uuid 
bin/kafka-storage.sh format \
  --cluster-id <PASTE_UUID_HERE> \
  --config /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server.properties

bin/kafka-storage.sh format \
  --cluster-id REdzqxscTZqoDAfMee9ktw \
  --config /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server.properties

```

## Make new KRaft instance: =====================================================

```bash
cd /mnt/nas_drive/achristopoulos/kafka-local/config/kraft
cp server.properties server-heavy.properties

log.dirs=/mnt/nas_drive/achristopoulos/kafka-kraft/logs-heavy
mkdir -p /mnt/nas_drive/achristopoulos/kafka-kraft/logs-heavy
bin/kafka-storage.sh random-uuid

bin/kafka-storage.sh format \
  --cluster-id <PASTE_CLUSTER_ID_HERE> \
  --config /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server-heavy.properties

  bin/kafka-storage.sh format \
  --cluster-id hQwr5n_sTpGaBH4UAYHv3Q \
  --config /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server-heavy.properties

```

## Error with log.dirs =================================================================

log.dirs=/mnt/nas_drive/achristopoulos/kafka-kraft/logs => μιλαμε για μεσα σε αυτο εδω το .dir

What Kafka KRaft stores in log.dirs. In KRaft mode, log.dirs is not only “topic data”. 
  It also contains the cluster’s brain:
  - the cluster.id (unique id for the KRaft cluster)
  - this node’s node.id
  - the controller quorum metadata log (the log that drives elections + metadata)
  - epochs / incarnation ids used to prove “I am the current controller” and to prevent split-brain

In KRaft, only the currently elected controller is allowed to accept and commit certain controller operations.

NotControllerException: The active controller appears to be node 1 means:
“The node that received this request is not currently the active controller (or it can’t prove it is), so it refuses to handle the controller request.”

## .bashrc ==========================================================================

~/.bashrc is a configuration file for the Bash shell:
    - It lives in your home directory (~)
    - whenever you open a new terminal or SSH session, Bash reads ~/.bashrc and executes whatever is inside.
        => if bashrc has this line:
        ```bash
        export KERAS_HOME=/mnt/nas_drive/achristopoulos/.keras
        ```
        => then it will execute this line once a new terminal launches 
    - source ~/.bashrc # Reload bashrc (reexecute all the commands)

```bash

echo 'source /mnt/nas_drive/achristopoulos/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc

echo 'export KERAS_HOME=/mnt/nas_drive/achristopoulos/.keras' >> ~/.bashrc
echo 'export KAFKA_HOME=/mnt/nas_drive/achristopoulos/kafka-local' >> ~/.bashrc
echo 'export PATH="$KAFKA_HOME/bin:$PATH"' >> ~/.bashrc
echo 'export JAVA_HOME=/mnt/nas_drive/achristopoulos/jdks/jdk-17.0.18+8' >> ~/.bashrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.bashrc
echo 'export JAVACPP_CACHE_DIR=/mnt/nas_drive/achristopoulos/.javacpp' >> ~/.bashrc
echo 'export XDG_CACHE_HOME=/mnt/nas_drive/achristopoulos/.cache' >> ~/.bashrc
echo 'export MAVEN_USER_HOME=/mnt/nas_drive/achristopoulos/.m2' >> ~/.bashrc
echo 'export TFDS_DATA_DIR=/mnt/nas_drive/achristopoulos/tensorflow_datasets' >> ~/.bashrc
```

# ===============================================================================
# How to get Keras 2 .h5 files:

```bash

# Activate the enviroment =======================================================
conda activate /mnt/nas_drive/achristopoulos/venvs/tf215

# Build the enviroment ==========================================================
source /mnt/nas_drive/achristopoulos/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/nas_drive/achristopoulos/venvs/tf215
source /mnt/nas_drive/achristopoulos/venvs/tf215/bin/activate
python -m pip install -U pip
python -m pip install "tensorflow[and-cuda]==2.15.*"
python -m pip install \
  numpy \
  pandas \
  scikit-learn \
  python-dotenv \
  kafka-python \
  matplotlib \ 
  tensorflow-datasets

```

## Server Wellness log: ============================================================================================

```bash
run-parts /etc/update-motd.d/   # Resee it using this
```


## ========================================================================================
## Datasets Installs:

```bash
# SVHN (Street view house numbers):
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
```
Enviromental Variables Ubuntu:

 - When you type kafka-topics.sh, then Ubuntu looks in the PATH enviroment variable from right to left tries to match the file to the location.
 - This means when you execute this command, the file doesnt have to be in pwd

ds123f15@ds123f15-Nitro-AN515-57:~$ ^C
ds123f15@ds123f15-Nitro-AN515-57:~$ ssh achristopoulos@polytechnix.softnet.tuc.gr
achristopoulos@polytechnix.softnet.tuc.gr's password: 
Welcome to Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-92-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

  System information as of Fri Feb 27 04:55:49 PM UTC 2026

  System load:                      6.603515625
  Usage of /:                       86.2% of 877.18GB
  Memory usage:                     26%
  Swap usage:                       99%
  Temperature:                      31.0 C
  Processes:                        1151
  Users logged in:                  3
  IPv4 address for br-113cb6b8be10: 192.168.192.1
  IPv4 address for br-400e4d3de2a2: 192.168.240.1
  IPv4 address for br-432a0ba7132e: 172.30.0.1
  IPv4 address for br-57cf8a87930e: 172.19.0.1
  IPv4 address for br-5c641bd65b3c: 172.24.0.1
  IPv4 address for br-5e119f36ea0d: 172.31.0.1
  IPv4 address for br-6a3c198f962f: 172.27.0.1
  IPv4 address for br-769f9bde856e: 172.23.0.1
  IPv4 address for br-9759ed98ce96: 172.21.0.1
  IPv4 address for br-9a3757b4c864: 172.28.0.1
  IPv4 address for br-9c01870fa928: 192.168.224.1
  IPv4 address for br-9c20c56e9ab4: 172.20.0.1
  IPv4 address for br-a195a21a49d4: 172.26.0.1
  IPv4 address for br-aca37d0c7092: 172.25.0.1
  IPv4 address for br-b4176f0e85dc: 192.168.64.1
  IPv4 address for br-b535d3fe4715: 172.22.0.1
  IPv4 address for br-ba0750f5d1f5: 172.18.0.1
  IPv4 address for br-d9a1387af70b: 172.29.0.1
  IPv4 address for docker0:         172.17.0.1
  IPv4 address for eno8303:         147.27.14.250

  => / is using 86.2% of 877.18GB
  => There are 206 zombie processes.

 * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s
   just raised the bar for easy, resilient and secure K8s cluster deployment.

   https://ubuntu.com/engage/secure-kubernetes-at-the-edge

Expanded Security Maintenance for Applications is not enabled.

404 updates can be applied immediately.
303 of these updates are standard security updates.
To see these additional updates run: apt list --upgradable

94 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm

New release '24.04.4 LTS' available.
Run 'do-release-upgrade' to upgrade to it.



The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.

achristopoulos@polytechnix:~$ 

here is some info for you ... 



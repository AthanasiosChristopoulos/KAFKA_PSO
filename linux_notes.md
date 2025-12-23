Apps from built in App Store:
 - telegram
 - vscode
 
Apps using dpkg:
 - sudo dpkg -i <file_name>.deb
 - Chrome
 
sudo apt install gnome-shell-extension-manager
 - extension: Hide TopBar

# Stop Lid: =====================================================================

 - https://www.youtube.com/watch?v=NEpoh89MYnc
 - sudo nano /etc/systemd/logind.conf
 - HandleLidSwitch=ignore
 
sudo apt install terminator
sudo update-alternatives --config x-terminal-emulator

## Git: ======================================================================

sudo apt install git
git config --global user.name "AthanasiosChristopoulos"
git config --global user.email "athanasioschristopoulos61@gmail.com"
git config --list

## Code Packages: ======================================================================

 - sudo apt install -y openjdk-17-jdk
 - java -version
 - javac -version
 - Set $JAVA_HOME:
 	echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
	echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
	source ~/.bashrc	

 - sudo apt install -y maven
 - mvn -version

 - sudo apt install -y \
	  python3 \
	  python3-pip \
	  python3-venv \
	  build-essential \
	  curl \
	  git
	  
 - Python Packages:
	pip install numpy pandas scikit-learn tensorflow kafka-python python-dotenv --break-system-packages


 - python3 --version
   pip3 --version

 - sudo apt install docker.io
 - sudo systemctl enable docker
 - sudo systemctl status docker
 
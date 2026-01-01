
# Apps from built in App Store: =============================================================

 - telegram
 - vscode

# Apps using dpkg: =========================================================================

 - sudo dpkg -i <file_name>.deb
 - Chrome
 
# Apps using commands: =====================================================================

 - sudo apt install terminator
 - sudo update-alternatives --config x-terminal-emulator
 - sudo apt install vlc
 - sudo apt install rhythmbox
 - sudo apt install audacious
 - sudo apt install gnome-shell-extension-manager
	- extension: Hide TopBar
 - sudo apt install mpv
	- nano ~/.config/mpv/mpv.conf
	- osd-bar=no

 - sudo apt install mesa-utils

# Vscode: ====================================================================================

 - Extensions:
	- Docker Python Java Github 
 - Settings:
	- Ctrl + Shift + P / Preferences: Open Keyboard Shortcuts:
		- go back: Alt + Left



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
```bash
pip install numpy --break-system-packages
pip install pandas --break-system-packages
pip install scikit-learn --break-system-packages
pip install tensorflow --break-system-packages
pip install kafka-python --break-system-packages
pip install python-dotenv --break-system-packages
pip install matplotlib --break-system-packages

```

 - python3 --version
   pip3 --version


# Docker on Ubuntu (either with commands or with docker desktop): =======================================

 - docker desktop:
	- https://www.youtube.com/watch?v=FWWq83IGUgw

 - command line:

	```bash
	sudo apt update
	sudo apt install -y docker.io docker-compose-plugin
	sudo systemctl enable --now docker
	sudo usermod -aG docker $USER
	newgrp docker
	# logout and login
	groups 		# verify that you are in the docker group
	docker ps
	docker run --rm hello-world
	docker compose up
	```
	- sudo apt install docker.io
	- sudo systemctl enable docker
	- sudo systemctl status docker
	

## Helper Notes ==========================================================================================

 - Disks:
 	- lsblk

	- rsync -av --delete /home/ds123f15/Documents/vvv/ /media/ds123f15/4AC86F1FC86F0891/projects/vvv/

	- sudo mkdir -p /mnt/OUT
	- sudo mount -a
	- ls /mnt/OUT
	- sudo umount /mnt/OUT

	- sudo mkdir -p /mnt/win_test
	  sudo mount -o ro /dev/nvme0n1p3 /mnt/win_test

## GPU ==========================================================================================

```bash

# disable / delete gpu drivers
sudo systemctl disable nvidia-persistenced
sudo apt purge 'nvidia*'
sudo update-initramfs -u
sudo reboot

# blacklist => disable automatic installs if this driver
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot

lsmod | grep nouveau	# should return nothing after blacklisting

glxinfo -B | grep "renderer"	# glxinfo is a tool that reports information about OpenGL, your graphics driver, and GPU setup.
								# This tells you which GPU is actually being used for rendering (the CPU or the GPU).


```

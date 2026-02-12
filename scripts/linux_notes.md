
# Apps from built in App Store: =============================================================

 - telegram
 - vscode

# Apps using dpkg: =========================================================================

 - sudo dpkg -i <file_name>.deb
 - Chrome
 
# Tools / Apps using commands: =====================================================================

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
 - sudo apt install brightnessctl (brightnessctl info / brightnessctl set 100%)

# Misc Commands: ======================================================================================

 - Disable low battery notifications:
	- gsettings set org.gnome.desktop.notifications show-banners false		# disable all notifications
	- systemctl --user mask org.gnome.SettingsDaemon.Power.service
	- systemctl --user stop org.gnome.SettingsDaemon.Power.service


# Vscode: =============================================================================================

 - Extensions:
	- Docker Python Java Github 
 - Settings:
	- Ctrl + Shift + P / Preferences: Open Keyboard Shortcuts:
		- go back: Alt + Left

sudo apt install gnome-shell-extension-manager
 - extension: Hide TopBar

# Change Boot Order / Boot Priority: ============================================================

https://www.youtube.com/watch?v=gVw1OMB-D5A

```bash
sudo nano /etc/default/grub				# Change GRUB_DEFAULT= to 0 (Ubuntu) or 4 (Windows)
sudo update-grub
grep GRUB_DEFAULT /etc/default/grub		# check new boot priority
```
# Power Stuff: =====================================================================

# Stop Lid: =====================================================================

 - https://www.youtube.com/watch?v=NEpoh89MYnc
 - sudo nano /etc/systemd/logind.conf
	- HandleLidSwitch=ignore
	- HandleLidSwitchExternalPower=ignore
	
sudo apt install terminator
sudo update-alternatives --config x-terminal-emulator

# Stop turning off Screen: =====================================================================
```bash
mkdir -p ~/.config/autostart
nano ~/.config/autostart/disable-dpms.desktop
```
Paste this:
[Desktop Entry]
Type=Application
Name=Disable DPMS
Exec=sh -c "xset -dpms; xset s off; xset s noblank"
X-GNOME-Autostart-enabled=true

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
PYTHONHTTPSVERIFY=1 python3 -m pip install pyswarms --user --default-timeout=120 --break-system-packages

# Check for installation location:
python3 -c "import pyswarms, inspect, os; print('pyswarms:', os.path.dirname(pyswarms.__file__))"
# It will be in /home/ds123f15/.local/lib/python3.12/site-packages/
```

or 

Ubuntu uses python as system package. You dont want to break them "break-system-packages" with your own stuff.
```bash
python3 -m venv ~/.venvs/dev
source ~/.venvs/dev/bin/activate	# do this for every project
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
	

## Disks: ==========================================================================================

 	- lsblk	(figure out name of the drive, i.e. sda1)

	- rsync -av --delete /home/ds123f15/Documents/vvv/ /media/ds123f15/4AC86F1FC86F0891/projects/vvv/

	- sudo mkdir -p /mnt/OUT
	- sudo mount -a
	- ls /mnt/OUT
	- sudo umount /mnt/OUT

	- sudo mkdir -p /mnt/win_test
	  sudo mount -o ro /dev/nvme0n1p3 /mnt/win_test

	=============================================================================================================
	- sudo apt update
	  sudo apt install ntfs-3g
	  sudo ntfsfix /dev/sda1
	  sudo mount /dev/sda1 /mnt
	  sudo umount /mnt

## GPU ==========================================================================================

```bash

# disable / delete gpu drivers ========================================
sudo systemctl disable nvidia-persistenced
sudo apt purge 'nvidia*'
sudo update-initramfs -u
sudo reboot

# # blacklist => disable automatic installs if this driver
# echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
# sudo update-initramfs -u
# sudo reboot

lsmod | grep nouveau	# should return nothing after blacklisting

glxinfo -B | grep "renderer"	# glxinfo is a tool that reports information about OpenGL, your graphics driver, and GPU setup.
								# This tells you which GPU is actually being used for rendering (the CPU or the GPU).

# ====================================================================
# prime => Linux graphics switching framework. sets which GPU is responsible for rendering and display (gpu or cpu)
sudo prime-select intel			# Only 
reboot

sudo prime-select nvidia	
reboot

sudo prime-select on-demand		# Intel display but CUDA works
sudo reboot

# Check which GPU is rendering: ======================================
prime-select query
# htop Alternatives for GPU:
nvidia-smi	# Confirm if it was successfully installed
glxinfo | grep "OpenGL renderer"


## Reinstall Driver: ===============================================

sudo apt update
sudo apt install nvidia-driver-535 nvidia-utils-535
sudo reboot

```

# Firefox: =====================================================================================
Remove bottom left panel showing the URL:
```bash
nano ~/snap/firefox/common/.mozilla/firefox/r382k5vg.default/chrome/userChrome.css
```
```css
#statuspanel {
display: none !important;
}
```


# Performance stuff: =====================================================================================

Create a systemd service

```bash
sudo nano /etc/systemd/system/cpu-performance.service

# Inside paste:

[Unit]
Description=Set CPU governor to performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/cpupower frequency-set -g performance
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target

# Enable it:
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable cpu-performance.service
sudo systemctl start cpu-performance.service

# Verify:
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Output:
performance

# Disable turbo:
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
cat /sys/devices/system/cpu/intel_pstate/no_turbo	# This needs to output 1
```
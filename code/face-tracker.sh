#! /bin/bash
source /home/pi/openvino/bin/setupvars.sh
export WORKON_HOME=/home/pi/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
VIRTUALENVWRAPPER_ENV_BIN_DIR=bin
workon openvino
cd /home/pi/face-tracker
/usr/bin/python3 runheadless.py -a face-tracker.py -f exitflag > face-tracker.log 2>&1

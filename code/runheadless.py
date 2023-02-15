#! /usr/bin/python3
#
# runheadless.py:
#
# Launches a headless application, provides status via GPIO outputs connected to LEDs,
# and monitors GPIO inputs to start or stop the application or shut down the system.
# A control panel of switches, buttons, and LEDs must be connected as defined in the
# physical pin mappings below.

import RPi.GPIO as GPIO
import re
import time
import os
import subprocess
import argparse

PIN_APPCTL  = 33    # button 1 pressed: start/stop application (toggle)
PIN_SHDNCTL = 31    # button 2 pressed: request system shutdown
PIN_APPENA  = 35    # switch 1 ON: application start/stop button enabled
PIN_SHDNENA = 36    # switch 2 ON: system shutdown button enabled
PIN_SYSRUN  = 40    # LED flashing: system is running
PIN_APPRUN  = 37    # LED flashing: application is running
PIN_SHDNSTS = 38    # LED ON: request system shutdown

def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def procExists(procName):
    """
    Check if a process having the given process name exists.  Adapted from post by
    Maksym Kozlenko on Stack Overflow on 08/10/2011.

    Parameters:
      procName - name of process to check

    Returns:
      exists - if True, a process with the given name exists
    """
    exists = False
    appPid = 0
    ps = subprocess.Popen("ps ax -o pid= -o args= ", shell=True, stdout=subprocess.PIPE)
    pspid = ps.pid
    procList = ps.stdout.read().decode('utf-8')
    ps.stdout.close()
    ps.wait()
    for line in procList.split("\n"):
        procSts = re.findall("(\d+) (.*)", line)
        if procSts:
            pid = int(procSts[0][0])
            if procName in procSts[0][1] and pid != os.getpid() and pid != pspid:
                exists = True
                appPid = pid
    return (exists, appPid)


###################################################################################
# MAIN program

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--app', required=True, help='file name of python script to run')
ap.add_argument('-f', '--flag', required=False, help='path to application exit flag file')
args = vars(ap.parse_args())

# Set up GPIO inputs and outputs.  Note that all inputs are active low.
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(PIN_SYSRUN, GPIO.OUT)
GPIO.setup(PIN_APPRUN, GPIO.OUT)
GPIO.setup(PIN_SHDNSTS, GPIO.OUT)
GPIO.output(PIN_SYSRUN, GPIO.LOW)
GPIO.output(PIN_APPRUN, GPIO.LOW)
GPIO.output(PIN_SHDNSTS, GPIO.LOW)
GPIO.setup(PIN_SHDNCTL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_SHDNENA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_APPCTL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_APPENA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(PIN_SHDNCTL, GPIO.FALLING, bouncetime=300)
GPIO.add_event_detect(PIN_APPCTL, GPIO.FALLING, bouncetime=300)

# These parameters control the period and duty cycle for flashing the status LEDs.
FLASH_PERIOD_SECS = 1.0
MAX_COUNT = 10
ON_COUNT  = 0   # count at which to turn LED on
OFF_COUNT = 1   # count at which to turn LED off
intervalSecs = FLASH_PERIOD_SECS / MAX_COUNT

appPid = 0
shutdownEnabled = False
reqShutdown = False
appEnabled = False
appRunning = False
GPIO.output(PIN_SYSRUN, GPIO.HIGH)

# Continuously report status until shutdown requested.
while reqShutdown == False:

    # Check switch status (note active low logic).
    if GPIO.input(PIN_APPENA) == 1 and appEnabled == True:
        print(f'{timestamp()} Switch 1 is OFF (application start/stop button disabled).')
        appEnabled = False
    elif GPIO.input(PIN_APPENA) == 0 and appEnabled == False:
        print(f'{timestamp()} Switch 1 is ON (application start/stop button enabled).')
        appEnabled = True
    if GPIO.input(PIN_SHDNENA) == 1 and shutdownEnabled == True:
        print(f'{timestamp()} Switch 2 is OFF (system shutdown button disabled).')
        shutdownEnabled = False
    elif GPIO.input(PIN_SHDNENA) == 0 and shutdownEnabled == False:
        print(f'{timestamp()} Switch 2 is ON (system shutdown button enabled).')
        shutdownEnabled = True

    # Flash the application status and system status LEDs.  Note that this also
    # adds to the main loop delay.
    for count in range(0, MAX_COUNT):
        if count == ON_COUNT:
            GPIO.output(PIN_SYSRUN, GPIO.HIGH)
        elif count == OFF_COUNT:
            GPIO.output(PIN_SYSRUN, GPIO.LOW)
        if appRunning:
            if count == ON_COUNT:
                GPIO.output(PIN_APPRUN, GPIO.HIGH)
            elif count == OFF_COUNT:
                GPIO.output(PIN_APPRUN, GPIO.LOW)
        time.sleep(intervalSecs)

    # Start/stop application if application start/stop button pressed.
    if GPIO.input(PIN_APPENA) == 0:   # 0 = app start/stop enable switch is ON
        if appEnabled == False:
            print(f'{timestamp()} Application start/stop button is now enabled.')
            appEnabled = True
            GPIO.event_detected(PIN_APPCTL)
        if GPIO.event_detected(PIN_APPCTL):
            (appRunning, appPid) = procExists(f'python3 {args["app"]}')
            if not appRunning:
                print(f'{timestamp()} Launching Python application {args["app"]}.')
                os.system(f'python3 {args["app"]} &')
                (appRunning, appPid) = procExists(f'python3 {args["app"]}')
                if appRunning:
                    print(f'{timestamp()} Successfully launched {args["app"]}, pid={appPid}.')
                else:
                    print(f'{timestamp()} Failed to launch {args["app"]}.')
            else:
                if args['flag'] is not None:
                    print(f'{timestamp()} Requested application exit {args["app"]}.')
                    os.system(f'ls > {args["flag"]}')
                    for checks in range(0, 12):
                        GPIO.output(PIN_APPRUN, GPIO.HIGH)
                        time.sleep(0.2)
                        GPIO.output(PIN_APPRUN, GPIO.LOW)
                        time.sleep(0.2)
                        (appRunning, appPid) = procExists(f'python3 {args["app"]}')
                        if not appRunning:
                            print(f'{timestamp()} Application {args["app"]} exited successfully.')
                            appRunning = False
                            break
                if appRunning:
                    print(f'{timestamp()} Killing application {args["app"]}, pid={appPid}.')
                    os.system(f'kill -9 {appPid}')
                    appRunning = False
            time.sleep(1)
    else:
        if appEnabled == True:
            print(f'{timestamp()} Application start/stop button is now disabled.')
            appEnabled = False

    # Prepare for shutdown if shutdown button pressed.
    if GPIO.input(PIN_SHDNENA) == 0:   # 0 = shutdown enable switch is ON
        if shutdownEnabled == False:
            print(f'{timestamp()} System shutdown button is now enabled.')
            shutdownEnabled = True
            GPIO.event_detected(PIN_SHDNCTL)
        if GPIO.event_detected(PIN_SHDNCTL):
            GPIO.output(PIN_SHDNSTS, GPIO.HIGH)
            reqShutdown = True
    else:
        if shutdownEnabled == True:
            print(f'{timestamp()} System shutdown button is now disabled.')
            shutdownEnabled = False

# If flagged, process the shutdown request.  Note that reqShutdown == True is the
# condition that causes the main loop to exit, but for safe code maintenance purposes,
# it will be tested explicitly here in case other loop exit conditions are added
# later.
if reqShutdown:
    if appRunning:
        print(f'{timestamp()} Killing application {args["app"]} process {appPid}...')
        os.system(f'kill -9 {appPid}')
    print(f'{timestamp()} Shutting down...')
    os.system('shutdown -h now')


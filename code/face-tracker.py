# face-tracker.py
#
# Track a face in a video stream and output servo controls to point
# a pair of anamatronic eyes at the face.  The eye hardware allows
# each eye to move independently, while the upper lids move in
# unison and the lower lids move in unison.
#
# Controls:
#   l - enter left eye adjustment mode, enabling numeric keypad control
#   r - enter right eye adjustment mode, enabling numeric keypad control
#   t - enter tracking mode, enabling automatic face tracking by both eyes
#   q - quit program and show final tracking positions
#   keypad 1 - capture lower left face detection servo position
#   keypad 3 - capture lower right face detection servo position
#   keypad 7 - capture upper left face detection servo position
#   keypad 9 - capture upper right face detection servo position
#   keypad 2 (down arrow) - adjust eyes down
#   keypad 4 (left arrow) - adjust eyes left
#   keypad 6 (right arrow) - adjust eyes right
#   keypad 8 (up arrow) - adjust eyes up
#

from os import _exit, system, getcwd, path, remove
import addpaths
import pca9685
from imutils.video import VideoStream
import numpy as np
import imutils
import math
import time
import cv2
import subprocess
import argparse
import RPi.GPIO as GPIO

#####################################################################
# CONFIGURATION PARAMETERS
#####################################################################

servoCfgFile = 'servo.conf'
modelFile = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxtFile = 'deploy.prototxt'
screamFile = 'scream1.wav'
laughFile = 'laugh.wav'
biteFile = 'bite.wav'
watchingFile = 'watching.wav'
exitFile = 'exitflag'
CONF_THRESH = 0.5           # confidence threshold for face detection
SERVO_FREQ_HZ = 50
CONTROL_I2C_ADDR = 0x40
MAX_NO_ACTIVITY_TIME = 10

#####################################################################
# GLOBAL CONSTANTS
#####################################################################

# Key assignments (returned by cv2.waitKey)
KEY_LOWER_L  = 177  # keypad 1
KEY_DOWN     = 178  # keypad 2
KEY_LOWER_R  = 179  # keypad 3
KEY_LEFT     = 180  # keypad 4
KEY_CENTER   = 181  # keypad 5
KEY_RIGHT    = 182  # keypad 6
KEY_UPPER_L  = 183  # keypad 7
KEY_UP       = 184  # keypad 8
KEY_UPPER_R  = 185  # keypad 9
KEY_ADJUST   = ord('a')
KEY_ADJUST_R = ord('r')
KEY_ADJUST_L = ord('l')
KEY_TRACK    = ord('t')
KEY_QUIT     = ord('q')

# These are the measured limits of the bounding box centroids and the corresponding
# number of pixels between the limits.
BB_MIN_X = 35
BB_MAX_X = 370
BB_MIN_Y = 35
BB_MAX_Y = 265

BB_WIDTH_X = BB_MAX_X - BB_MIN_X   # width in pixels of area where face can be found
BB_HEIGHT_Y = BB_MAX_Y - BB_MIN_Y  # height in pixels of area where face can be found
BB_AREA_RELAXED = 6000             # size of bounding box to trigger relaxed behavior
BB_AREA_STARTLED = 32000           # size of bounding box to trigger startled behavior


#####################################################################
# CLASSES
#####################################################################

class EyeControl():
    """
    This class abstracts the controller setting of eye positions to be relative
    to the configured range of horizontal and vertical angles of the servos.
    """

    MOVE_UP    = 1
    MOVE_DOWN  = 2
    MOVE_LEFT  = 3
    MOVE_RIGHT = 4

    def __init__(self, controller, leftOrRight):
        self.controller = controller
        self.leftOrRight = leftOrRight
        self.chanH = None
        self.chanV = None
        self.chanAngleH = pca9685.ANGLE_INVALID
        self.chanAngleV = pca9685.ANGLE_INVALID
        self.minAngleH = None
        self.maxAngleH = None
        self.minAngleV = None
        self.maxAngleV = None
        self.degreesH = None
        self.degreesV = None
        # Dictionary of recorded face position locations.
        self.locsDict = { KEY_LOWER_L : (0.0, 0.0),
                          KEY_LOWER_R : (0.0, 0.0),
                          KEY_UPPER_L : (0.0, 0.0),
                          KEY_UPPER_R : (0.0, 0.0),
                        }
        # Marked location key names
        self.recordDict = { KEY_LOWER_L : 'lower L',
                            KEY_LOWER_R : 'lower R',
                            KEY_UPPER_L : 'upper L',
                            KEY_UPPER_R : 'upper R'
                           }



    def setConfig(self):
        """
        Set the channel numbers and angle limits for the eyeball servos from their
        configuration names.
        """
        isValid = False
        if self.leftOrRight == 'left':
            self.chanH = self.controller.getChannel('left-x-axis')
            self.chanV = self.controller.getChannel('left-y-axis')
            (self.minAngleH, self.maxAngleH) = self.controller.getLimits('left-x-axis')
            (self.minAngleV, self.maxAngleV) = self.controller.getLimits('left-y-axis')
        else:
            self.chanH = self.controller.getChannel('right-x-axis')
            self.chanV = self.controller.getChannel('right-y-axis')
            (self.minAngleH, self.maxAngleH) = self.controller.getLimits('right-x-axis')
            (self.minAngleV, self.maxAngleV) = self.controller.getLimits('right-y-axis')
        if (not self.chanH == None and not self.chanV == None and
            not self.minAngleH == None and not self.maxAngleH == None and
            not self.minAngleV == None and not self.maxAngleV == None):
            self.degreesH = self.maxAngleH - self.minAngleH
            self.degreesV = self.maxAngleV - self.minAngleV
            isValid = True
        print(f'{self.leftOrRight} eye servo configuration:')
        print(f'  x-axis chan={self.chanH}, min angle={self.minAngleH}, max angle={self.maxAngleH}')
        print(f'  y-axis chan={self.chanV}, min angle={self.minAngleV}, max angle={self.maxAngleV}')
        return isValid

    def center(self):
        """
        Center the eye horizontally and vertically.
        """
        self.chanAngleH = self.controller.setServoPosition(self.chanH, 'center')
        self.chanAngleV = self.controller.setServoPosition(self.chanV, 'center')

    def adjust(self, direction, change):
        """
        Apply a change in degrees to the horizontal or vertical position of the eyeball.
        """
        angleNewH = pca9685.ANGLE_INVALID
        angleNewV = pca9685.ANGLE_INVALID
        if direction == EyeControl.MOVE_UP:
            if self.leftOrRight == 'right':
                change = -change
            angleNewV = self.controller.setServoAngle(self.chanV, self.chanAngleV + change)
        elif direction == EyeControl.MOVE_DOWN:
            if self.leftOrRight == 'right':
                change = -change
            angleNewV = self.controller.setServoAngle(self.chanV, self.chanAngleV - change)
        elif direction == EyeControl.MOVE_LEFT:
            angleNewH = self.controller.setServoAngle(self.chanH, self.chanAngleH + change)
        elif direction == EyeControl.MOVE_RIGHT:
            angleNewH = self.controller.setServoAngle(self.chanH, self.chanAngleH - change)
        if not angleNewH == pca9685.ANGLE_INVALID:
            self.chanAngleH = angleNewH
        if not angleNewV == pca9685.ANGLE_INVALID:
            self.chanAngleV = angleNewV

    def setRelativePosition(self, ratioHoriz, ratioVert):
        """
        Set the relative horizontal (0 - 1.0) and vertical (0 - 1.0) position of
        the eyeball, where a horizontal ratio of 1.0 is all the way to the left and
        a vertical ratio of 1.0 is all the way up.
        """
        angleNewH = self.minAngleH + (ratioHoriz * self.degreesH)
        if self.leftOrRight == 'left':
            angleNewV = self.minAngleV + (ratioVert * self.degreesV)
        else:
            angleNewV = self.maxAngleV - (ratioVert * self.degreesV)
        angleNewH = self.controller.setServoAngle(self.chanH, angleNewH)
        angleNewV = self.controller.setServoAngle(self.chanV, angleNewV)
        if not angleNewH == pca9685.ANGLE_INVALID:
            self.chanAngleH = angleNewH
        if not angleNewV == pca9685.ANGLE_INVALID:
            self.chanAngleV = angleNewV

    def getCurrentPosition(self):
        """
        Return current horizontal and vertical eyeball servo positions in degrees.
        """
        return (self.chanAngleH, self.chanAngleV)

    def markLocation(self, cmdKey):
        self.locsDict[cmdKey] = (self.chanAngleH, self.chanAngleV)
        print(f'Captured {self.recordDict[cmdKey]} position at {self.locsDict[cmdKey]}.')


class ExpressionControl():
    """
    This class abstracts the controller setting of eyelid positions, jaw position, and
    audio effects as behaviors.
    """

    ASLEEP = 1
    RELAXED = 2
    STARTLED = 3
    SUSPICIOUS = 4
    LAUGHING = 5
    WIDEEYED = 6
    BITING = 7

    def __init__(self, controller):
        self.controller = controller
        self.chLidUpper = None
        self.chLidLower = None
        self.chJaw = None
        self.current = None

    def get(self):
        return self.current

    def config(self):
        """
        Set the channel numbers for each servo from their configuration names.
        """
        self.chLidUpper = self.controller.getChannel('upper-lids')
        self.chLidLower = self.controller.getChannel('lower-lids')
        self.chJaw       = self.controller.getChannel('jaw')
        isValid = (not self.chLidUpper == None and
                   not self.chLidLower == None
                   and not self.chJaw == None)
        print('Configure channels:')
        print(f'  upper-lids={self.chLidUpper}')
        print(f'  lower-lids={self.chLidLower}')
        print(f'  jaw={self.chJaw}')
        return isValid

    def setAsleep(self):
        self.current = ExpressionControl.ASLEEP
        self.controller.setServoPosition(self.chLidUpper, 'closed')
        self.controller.setServoPosition(self.chLidLower, 'closed')
        self.controller.setServoPosition(self.chJaw, 'closed')

    def setRelaxed(self):
        self.current = ExpressionControl.RELAXED
        self.controller.setServoPosition(self.chLidUpper, 'halfopen')
        self.controller.setServoPosition(self.chLidLower, 'halfopen')
        self.controller.setServoPosition(self.chJaw, 'closed')

    def setWideEyed(self):
        self.current = ExpressionControl.WIDEEYED
        self.controller.setServoPosition(self.chLidUpper, 'open')
        self.controller.setServoPosition(self.chLidLower, 'open')
        self.controller.setServoPosition(self.chJaw, 'closed')

    def setStartled(self, soundFile):
        self.current = ExpressionControl.STARTLED
        soundProc = subprocess.Popen(['aplay', '-Dhw:0,0', soundFile])
        returnCode = soundProc.poll()
        if returnCode != None:
            print(f'ERROR: Launch of aplay failed with return code {returnCode}.')
        self.controller.setServoPosition(self.chLidUpper, 'open')
        self.controller.setServoPosition(self.chLidLower, 'open')
        self.controller.setServoPosition(self.chJaw, 'open')

    def setSuspicious(self, fraction):
        self.current = ExpressionControl.SUSPICIOUS
        self.controller.setServoPositionRel(self.chLidUpper, 'halfopen', 'closed', fraction)
        self.controller.setServoPositionRel(self.chLidLower, 'halfopen', 'closed', fraction)
        self.controller.setServoPosition(self.chJaw, 'closed')

    def setLaughing(self, soundFile):
        self.current = ExpressionControl.LAUGHING
        soundProc = subprocess.Popen(['aplay', '-Dhw:0,0', soundFile])
        returnCode = soundProc.poll()
        if returnCode != None:
            print(f'ERROR: Launch of aplay failed with return code {returnCode}.')
        for i in range(0,6):
            self.controller.setServoPositionRel(self.chJaw, 'closed', 'open', 0.2)
            time.sleep(0.1)
            self.controller.setServoPosition(self.chJaw, 'closed')
            time.sleep(0.1)

    def setWatching(self, soundFile):
        self.current = ExpressionControl.LAUGHING
        soundProc = subprocess.Popen(['aplay', '-Dhw:0,0', soundFile])
        returnCode = soundProc.poll()
        if returnCode != None:
            print(f'ERROR: Launch of aplay failed with return code {returnCode}.')
        self.controller.setServoPosition(self.chLidUpper, 'open')
        self.controller.setServoPosition(self.chLidLower, 'open')
        # we're
        self.controller.setServoPositionRel(self.chJaw, 'closed', 'open', 0.3)
        time.sleep(0.8)
        self.controller.setServoPosition(self.chJaw, 'closed')
        time.sleep(0.15)
        # watch-
        self.controller.setServoPositionRel(self.chJaw, 'closed', 'open', 0.5)
        time.sleep(0.2)
        self.controller.setServoPosition(self.chJaw, 'closed')
        time.sleep(0.1)
        # -ing
        self.controller.setServoPositionRel(self.chJaw, 'closed', 'open', 0.3)
        time.sleep(0.2)
        self.controller.setServoPosition(self.chJaw, 'closed')
        time.sleep(0.1)
        # you
        self.controller.setServoPositionRel(self.chJaw, 'closed', 'open', 0.5)
        time.sleep(0.8)
        self.controller.setServoPosition(self.chJaw, 'closed')
        time.sleep(0.1)

    def setBiting(self, soundFile):
        self.current = ExpressionControl.BITING
        soundProc = subprocess.Popen(['aplay', '-Dhw:0,0', soundFile])
        returnCode = soundProc.poll()
        if returnCode != None:
            print(f'ERROR: Launch of aplay failed with return code {returnCode}.')
        self.controller.setServoPosition(self.chLidUpper, 'open')
        self.controller.setServoPosition(self.chLidLower, 'open')
        self.controller.setServoPositionRel(self.chJaw, 'closed', 'open', 0.4)
        time.sleep(0.1)
        self.controller.setServoPosition(self.chJaw, 'closed')
        time.sleep(0.1)


class Lighting():
    """
    This class encapsulates the GPIO outputs used to turn lighting on and off.
    """

    PIN_LIGHT1 = 13
    PIN_LIGHT2 = 15
    PIN_LIGHT3 = 16

    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(Lighting.PIN_LIGHT1, GPIO.OUT)
        GPIO.setup(Lighting.PIN_LIGHT2, GPIO.OUT)
        GPIO.setup(Lighting.PIN_LIGHT3, GPIO.OUT)

    def on(self):
        GPIO.output(Lighting.PIN_LIGHT1, GPIO.HIGH)
        GPIO.output(Lighting.PIN_LIGHT2, GPIO.HIGH)
        GPIO.output(Lighting.PIN_LIGHT3, GPIO.HIGH)

    def off(self):
        GPIO.output(Lighting.PIN_LIGHT1, GPIO.LOW)
        GPIO.output(Lighting.PIN_LIGHT2, GPIO.LOW)
        GPIO.output(Lighting.PIN_LIGHT3, GPIO.LOW)


#####################################################################
# MAIN
#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--display', required=False, action='store_true',
                    help='display video')
parser.add_argument('-b', '--box', required=False, action='store_true',
                    help='draw box around face')
args = vars(parser.parse_args())
showImage = False           # display captured frame when True
drawBox = False             # draw box around detected face when True
if args['display'] == True:
    showImage = True
if args['box'] == True:
    drawBox = True

exitPath = path.join(getcwd(), exitFile)

# Adjust key names
adjustDict = {KEY_DOWN   : 'down',
              KEY_LEFT   : 'left',
              KEY_CENTER : 'center',
              KEY_RIGHT  : 'right',
              KEY_UP     : 'up'
             }

# Mode key names
modeDict = {KEY_TRACK  : 'track',
            KEY_ADJUST : 'adjust',
            KEY_ADJUST_R: 'adjust right',
            KEY_ADJUST_L: 'adjust left'
           }

# Initialize the servo controller and load its configuration.
print('Initializing servo controller...')
controller = pca9685.PCA9685Controller(CONTROL_I2C_ADDR)
controlReady = controller.loadConfig(servoCfgFile)
if not controlReady:
    print('ERROR: Unable to load controller configuration.  Quitting.')
    _exit(1)
controller.start(SERVO_FREQ_HZ)
controller.printStatus()

# Create and initialize the eye and expression control objects.  Note that the
# min/max horizontal/vertical angle limits of the eye servos are determined
# empirically by adjusting the eye positions so that they appear to point at
# a face detected at each edge of the frame.
expression = ExpressionControl(controller)
isExpCfgValid = expression.config()
if not isExpCfgValid:
    print('ERROR: Configuration of ExpressionControl failed.  Quitting.')
    _exit(1)
expression.setAsleep()
isStartleArmed = True
eyeL = EyeControl(controller, 'left')
eyeR = EyeControl(controller, 'right')
isEyeCfgValid = eyeL.setConfig() and eyeR.setConfig()
if not isEyeCfgValid:
    print('ERROR: Configuration of EyeControl failed.  Quitting.')
    _exit(1)
eyeL.center()
eyeR.center()
activeEye = eyeL
mode = modeDict[KEY_TRACK]

# Create the lighting control and turn on the lights.
lights = Lighting()
lights.on()

# Load the serialized model from disk.
print('Loading model...')
net = cv2.dnn.readNetFromCaffe(prototxtFile, modelFile)

# Use the Raspberry Pi hardware for the inference engine.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize the video stream and allow the camera sensor to warm up.
print('Starting video stream...')
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
print(f'Initialization complete, entering {mode} mode.')

# Loop over the frames from the video stream.
timeLastActive = time.time()
while True:

    # Check for keyboard input to change operational mode, adjust position,
    # record position, or quit.
    cmdKey = cv2.waitKey(1) & 0xff
    if not cmdKey == 0xff:
        if cmdKey == KEY_QUIT:
            break
        if cmdKey in modeDict.keys():
            if not mode == modeDict[cmdKey]:
                print(f'Mode changed from {mode} to {modeDict[cmdKey]}.')
                mode = modeDict[cmdKey]
                if mode == modeDict[KEY_ADJUST_R]:
                    activeEye = eyeR
                    expression.setWideEyed()
                elif mode == modeDict[KEY_ADJUST_L]:
                    activeEye = eyeL
                    expression.setWideEyed()
                else:
                    expression.setAsleep()
            else:
                print(f'Already in {mode} mode.')

        # Apply adjustment to eye position if in adjustment mode.
        if (cmdKey in adjustDict.keys() and
            (mode == modeDict[KEY_ADJUST_R] or mode == modeDict[KEY_ADJUST_L])):
            if cmdKey == KEY_CENTER:
                activeEye.center()
            elif cmdKey == KEY_UP:
                activeEye.adjust(EyeControl.MOVE_UP, 3.0)
            elif cmdKey == KEY_DOWN:
                activeEye.adjust(EyeControl.MOVE_DOWN, 3.0)
            elif cmdKey == KEY_RIGHT:
                activeEye.adjust(EyeControl.MOVE_RIGHT, 3.0)
            elif cmdKey == KEY_LEFT:
                activeEye.adjust(EyeControl.MOVE_LEFT, 3.0)
        if (cmdKey in activeEye.locsDict.keys() and
            (mode == modeDict[KEY_ADJUST_R] or mode == modeDict[KEY_ADJUST_L])):
            activeEye.markLocation(cmdKey)

    try:

        # Grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels.
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        frame = imutils.rotate(frame, angle=180)

        # Grab the frame dimensions and convert it to a blob.
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections and predictions.
        startTime = time.time()
        net.setInput(blob)
        detections = net.forward()
        detectTime = time.time() - startTime

        # Loop over the detections
        for i in range(0, detections.shape[2]):

            # Extract the confidence (i.e., probability) associated with the prediction.
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the `confidence` is greater than
            # the minimum confidence.
            if confidence < CONF_THRESH:
                continue

            # Compute the (x, y)-coordinates of the bounding box centroid for the object.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            centerX = (endX + startX) // 2
            centerY = (endY + startY) // 2
            boxArea = (endX - startX) * (endY - startY)

            # Compute the relative position of the bounding box centroid.
            ratioH = 1.0 - (centerX - BB_MIN_X) / BB_WIDTH_X
            if ratioH < 0.0:
                ratioH = 0.0
            elif ratioH > 1.0:
                ratioH = 1.0
            ratioV = 1.0 - (centerY - BB_MIN_Y) / BB_HEIGHT_Y
            if ratioV < 0.0:
                ratioV = 0.0
            elif ratioV > 1.0:
                ratioV = 1.0

            # If in tracking mode, set the eye position and the expression according
            # to the position and size of the detected bounding box.
            if (mode == modeDict[KEY_TRACK] and time.time() > timeLastActive + 1.5):
                eyeL.setRelativePosition(ratioH, ratioV)
                eyeR.setRelativePosition(ratioH, ratioV)

                # When face is farthest, display relaxed expression, but if
                # currently asleep, wake up and display 'crazy-eyes' sequence
                # first.
                if boxArea < BB_AREA_RELAXED:
                    if expression.get() == ExpressionControl.ASLEEP:
                        expression.setWatching(watchingFile)
                        time.sleep(1.0)
                        for i in range(0,360,9):
                            hLeft = (1.0 + math.sin(math.radians(i)))/2.0
                            vLeft = (1.0 + math.cos(math.radians(i)))/2.0
                            hRight = (1.0 + math.sin(math.radians(i+180)))/2.0
                            vRight = (1.0 + math.cos(math.radians(i+180)))/2.0
                            eyeL.setRelativePosition(hLeft, vLeft)
                            eyeR.setRelativePosition(hRight, vRight)
                            time.sleep(0.02)
                        time.sleep(0.5)
                        eyeL.center()
                        eyeR.center()
                        time.sleep(1.0)
                    isStartleArmed = True
                    expression.setRelaxed()

                # When face is closest, scream and laugh.
                elif boxArea > BB_AREA_STARTLED and isStartleArmed:
                    isStartleArmed = False
                    expression.setBiting(biteFile)
                    time.sleep(2)
                    expression.setRelaxed()

                # When face is neither far or close, display suspicious expression based on
                # distance.
                elif isStartleArmed:
                    fraction = (boxArea - BB_AREA_RELAXED) / (BB_AREA_STARTLED - BB_AREA_RELAXED)
                    expression.setSuspicious(fraction)

                timeLastActive = time.time()

            # Log data about current position of eye and face detected.
            (horizPosL, vertPosL) = eyeL.getCurrentPosition()
            (horizPosR, vertPosR) = eyeR.getCurrentPosition()
#            print(f'face at {ratioH:.3f},{ratioV:.3f} '
#                  f'(HL {horizPosL:.1f} VL {vertPosL:.1f} '
#                  f'HR {horizPosR:.1f} VR {vertPosR:.1f} '
#                  f'{mode}); {detectTime:.4f} sec; area {boxArea}')

            # Draw the bounding box of the face along with the associated probability.
            if showImage == True and drawBox == True:
                text = '{:.2f}%'.format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255), 2)

            # Detected at least one good face -- ignore others and exit loop.
            break

        # end of detection loop

        # If no faces have been detected recently in tracking mode, set the expression to
        # asleep.
        if timeLastActive != None:
            idleTime = time.time() - timeLastActive
            if idleTime >= MAX_NO_ACTIVITY_TIME and mode == modeDict[KEY_TRACK]:
                expression.setAsleep()

        # Show the output frame.
        if showImage == True:
            cv2.imshow('Frame', frame)

    except KeyboardInterrupt:
        break

    # Presence of exit flag file causes application to exit.
    if path.exists(exitPath):
        remove(exitPath)
        break

# Cleanup before exit.
print('\nUser requested exit.  Cleaning up...')
expression.setRelaxed()
eyeL.center()
eyeR.center()
lights.off()
cv2.destroyAllWindows()
vs.stop()
print('\nFinal recorded face positions for left eye adjustment:')
for (cmdKey, location) in eyeL.locsDict.items():
    print(f'{eyeL.recordDict[cmdKey]}  {location[0]} {location[1]}')
print('\nFinal recorded face positions for right eye adjustment:')
for (cmdKey, location) in eyeR.locsDict.items():
    print(f'{eyeR.recordDict[cmdKey]}  {location[0]} {location[1]}')

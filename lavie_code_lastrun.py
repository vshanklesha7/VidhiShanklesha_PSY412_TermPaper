#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.3),
    on November 11, 2026, at 21:45
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.3'
expName = 'lavie_code'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = 'data/' + expInfo['participant'] + '_' + expName + '_' + expInfo['date']
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\shank\\OneDrive\\Desktop\\PSY412\\Experiement Code_Lavie (1995)\\lavie_code_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    text = visual.TextStim(win=win, name='text',
        text='You will see a target letter in the centre of the screen, along with another letter above or below it.\n\nIdentify the target letter in the center of the display.\n\nPress 2 when the target is X.\n\nPress 0 when the target is Z.\n\nRespond as quickly and accurately as possible.\n\nThere will be a few practice trials first, and then the main experiment will begin!\n\nPress the SPACEBAR to begin.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "pilot" ---
    pilot_fixation = visual.ShapeStim(
        win=win, name='pilot_fixation', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    letter_1 = visual.TextStim(win=win, name='letter_1',
        text='',
        font='Arial',
        pos=(-0.30, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    letter_2 = visual.TextStim(win=win, name='letter_2',
        text='',
        font='Arial',
        pos=(-0.18, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    letter_3 = visual.TextStim(win=win, name='letter_3',
        text='',
        font='Arial',
        pos=(-0.06, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    letter_4 = visual.TextStim(win=win, name='letter_4',
        text='',
        font='Arial',
        pos=(0.06, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    letter_5 = visual.TextStim(win=win, name='letter_5',
        text='',
        font='Arial',
        pos=(0.18, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    letter_6 = visual.TextStim(win=win, name='letter_6',
        text='',
        font='Arial',
        pos=(0.30, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    peripheral_distractor = visual.TextStim(win=win, name='peripheral_distractor',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    pilot_response = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "main_instructions" ---
    instructions_main = visual.TextStim(win=win, name='instructions_main',
        text='You are now done with the practice trials!\n\nThe main experiment will consist of 180 trials. \n\nThere will be a short break after 90 trials. \n\nThe instructions remain the same for the main experiement\nPress 2 when target is X\nPress 0 when target is Z\n\nPress SPACE whenever you are ready to start with the experiment!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start_experiment = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "trial" ---
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    letter1 = visual.TextStim(win=win, name='letter1',
        text='',
        font='Arial',
        pos=(-0.30, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    letter2 = visual.TextStim(win=win, name='letter2',
        text='',
        font='Arial',
        pos=(-0.18, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    letter3 = visual.TextStim(win=win, name='letter3',
        text='',
        font='Arial',
        pos=(-0.06, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    letter4 = visual.TextStim(win=win, name='letter4',
        text='',
        font='Arial',
        pos=(0.06, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    letter5 = visual.TextStim(win=win, name='letter5',
        text='',
        font='Arial',
        pos=(0.18, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    letter6 = visual.TextStim(win=win, name='letter6',
        text='',
        font='Arial',
        pos=(0.30, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    peripheralDistractor = visual.TextStim(win=win, name='peripheralDistractor',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    main_response = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "break_3min" ---
    break_instructions = visual.TextStim(win=win, name='break_instructions',
        text='You may now take a short break.\n\n\n\nThe experiment will continue automatically in 3 minutes.\n\n\n\nPlease remain seated and avoid closing the experiment.\n\nPress SPACE if you would like to continue immediately.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    break_text = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "trial" ---
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    letter1 = visual.TextStim(win=win, name='letter1',
        text='',
        font='Arial',
        pos=(-0.30, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    letter2 = visual.TextStim(win=win, name='letter2',
        text='',
        font='Arial',
        pos=(-0.18, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    letter3 = visual.TextStim(win=win, name='letter3',
        text='',
        font='Arial',
        pos=(-0.06, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    letter4 = visual.TextStim(win=win, name='letter4',
        text='',
        font='Arial',
        pos=(0.06, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    letter5 = visual.TextStim(win=win, name='letter5',
        text='',
        font='Arial',
        pos=(0.18, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    letter6 = visual.TextStim(win=win, name='letter6',
        text='',
        font='Arial',
        pos=(0.30, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    peripheralDistractor = visual.TextStim(win=win, name='peripheralDistractor',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    main_response = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "thank_you" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Thank you so much for your participation! \n\nPress SPACE to end the experiment. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[text, key_resp],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    thisExp.currentRoutine = instructions
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pilot_loop = data.TrialHandler2(
        name='pilot_loop',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('C:/Users/shank/OneDrive/Desktop/Lavie_1995_Pilot_18_Conditions.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(pilot_loop)  # add the loop to the experiment
    thisPilot_loop = pilot_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPilot_loop.rgb)
    if thisPilot_loop != None:
        for paramName in thisPilot_loop:
            globals()[paramName] = thisPilot_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPilot_loop in pilot_loop:
        pilot_loop.status = STARTED
        if hasattr(thisPilot_loop, 'status'):
            thisPilot_loop.status = STARTED
        currentLoop = pilot_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPilot_loop.rgb)
        if thisPilot_loop != None:
            for paramName in thisPilot_loop:
                globals()[paramName] = thisPilot_loop[paramName]
        
        # --- Prepare to start Routine "pilot" ---
        # create an object to store info about Routine pilot
        pilot = data.Routine(
            name='pilot',
            components=[pilot_fixation, letter_1, letter_2, letter_3, letter_4, letter_5, letter_6, peripheral_distractor, pilot_response],
        )
        pilot.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from pilot_code
        if pilot_distractor_pos == "above":
            distractor_position = (0, 0.25)
        else:
            distractor_position = (0, -0.25)
        letter_1.setText(pilot_pos_1)
        letter_2.setText(pilot_pos_2
        
        )
        letter_3.setText(pilot_pos_3)
        letter_4.setText(pilot_pos_4)
        letter_5.setText(pilot_pos_5)
        letter_6.setText(pilot_pos_6)
        peripheral_distractor.setPos(distractor_position)
        peripheral_distractor.setText(pilot_distractor)
        # create starting attributes for pilot_response
        pilot_response.keys = []
        pilot_response.rt = []
        _pilot_response_allKeys = []
        # store start times for pilot
        pilot.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pilot.tStart = globalClock.getTime(format='float')
        pilot.status = STARTED
        thisExp.addData('pilot.started', pilot.tStart)
        pilot.maxDuration = None
        # keep track of which components have finished
        pilotComponents = pilot.components
        for thisComponent in pilot.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pilot" ---
        thisExp.currentRoutine = pilot
        pilot.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # if trial has changed, end Routine now
            if hasattr(thisPilot_loop, 'status') and thisPilot_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pilot_fixation* updates
            
            # if pilot_fixation is starting this frame...
            if pilot_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pilot_fixation.frameNStart = frameN  # exact frame index
                pilot_fixation.tStart = t  # local t and not account for scr refresh
                pilot_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pilot_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pilot_fixation.started')
                # update status
                pilot_fixation.status = STARTED
                pilot_fixation.setAutoDraw(True)
            
            # if pilot_fixation is active this frame...
            if pilot_fixation.status == STARTED:
                # update params
                pass
            
            # if pilot_fixation is stopping this frame...
            if pilot_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > pilot_fixation.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    pilot_fixation.tStop = t  # not accounting for scr refresh
                    pilot_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    pilot_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pilot_fixation.stopped')
                    # update status
                    pilot_fixation.status = FINISHED
                    pilot_fixation.setAutoDraw(False)
            
            # *letter_1* updates
            
            # if letter_1 is starting this frame...
            if letter_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter_1.frameNStart = frameN  # exact frame index
                letter_1.tStart = t  # local t and not account for scr refresh
                letter_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter_1.started')
                # update status
                letter_1.status = STARTED
                letter_1.setAutoDraw(True)
            
            # if letter_1 is active this frame...
            if letter_1.status == STARTED:
                # update params
                pass
            
            # if letter_1 is stopping this frame...
            if letter_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter_1.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter_1.tStop = t  # not accounting for scr refresh
                    letter_1.tStopRefresh = tThisFlipGlobal  # on global time
                    letter_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter_1.stopped')
                    # update status
                    letter_1.status = FINISHED
                    letter_1.setAutoDraw(False)
            
            # *letter_2* updates
            
            # if letter_2 is starting this frame...
            if letter_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter_2.frameNStart = frameN  # exact frame index
                letter_2.tStart = t  # local t and not account for scr refresh
                letter_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter_2.started')
                # update status
                letter_2.status = STARTED
                letter_2.setAutoDraw(True)
            
            # if letter_2 is active this frame...
            if letter_2.status == STARTED:
                # update params
                pass
            
            # if letter_2 is stopping this frame...
            if letter_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter_2.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter_2.tStop = t  # not accounting for scr refresh
                    letter_2.tStopRefresh = tThisFlipGlobal  # on global time
                    letter_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter_2.stopped')
                    # update status
                    letter_2.status = FINISHED
                    letter_2.setAutoDraw(False)
            
            # *letter_3* updates
            
            # if letter_3 is starting this frame...
            if letter_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter_3.frameNStart = frameN  # exact frame index
                letter_3.tStart = t  # local t and not account for scr refresh
                letter_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter_3.started')
                # update status
                letter_3.status = STARTED
                letter_3.setAutoDraw(True)
            
            # if letter_3 is active this frame...
            if letter_3.status == STARTED:
                # update params
                pass
            
            # if letter_3 is stopping this frame...
            if letter_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter_3.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter_3.tStop = t  # not accounting for scr refresh
                    letter_3.tStopRefresh = tThisFlipGlobal  # on global time
                    letter_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter_3.stopped')
                    # update status
                    letter_3.status = FINISHED
                    letter_3.setAutoDraw(False)
            
            # *letter_4* updates
            
            # if letter_4 is starting this frame...
            if letter_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter_4.frameNStart = frameN  # exact frame index
                letter_4.tStart = t  # local t and not account for scr refresh
                letter_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter_4.started')
                # update status
                letter_4.status = STARTED
                letter_4.setAutoDraw(True)
            
            # if letter_4 is active this frame...
            if letter_4.status == STARTED:
                # update params
                pass
            
            # if letter_4 is stopping this frame...
            if letter_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter_4.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter_4.tStop = t  # not accounting for scr refresh
                    letter_4.tStopRefresh = tThisFlipGlobal  # on global time
                    letter_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter_4.stopped')
                    # update status
                    letter_4.status = FINISHED
                    letter_4.setAutoDraw(False)
            
            # *letter_5* updates
            
            # if letter_5 is starting this frame...
            if letter_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter_5.frameNStart = frameN  # exact frame index
                letter_5.tStart = t  # local t and not account for scr refresh
                letter_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter_5.started')
                # update status
                letter_5.status = STARTED
                letter_5.setAutoDraw(True)
            
            # if letter_5 is active this frame...
            if letter_5.status == STARTED:
                # update params
                pass
            
            # if letter_5 is stopping this frame...
            if letter_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter_5.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter_5.tStop = t  # not accounting for scr refresh
                    letter_5.tStopRefresh = tThisFlipGlobal  # on global time
                    letter_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter_5.stopped')
                    # update status
                    letter_5.status = FINISHED
                    letter_5.setAutoDraw(False)
            
            # *letter_6* updates
            
            # if letter_6 is starting this frame...
            if letter_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter_6.frameNStart = frameN  # exact frame index
                letter_6.tStart = t  # local t and not account for scr refresh
                letter_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter_6.started')
                # update status
                letter_6.status = STARTED
                letter_6.setAutoDraw(True)
            
            # if letter_6 is active this frame...
            if letter_6.status == STARTED:
                # update params
                pass
            
            # if letter_6 is stopping this frame...
            if letter_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter_6.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter_6.tStop = t  # not accounting for scr refresh
                    letter_6.tStopRefresh = tThisFlipGlobal  # on global time
                    letter_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter_6.stopped')
                    # update status
                    letter_6.status = FINISHED
                    letter_6.setAutoDraw(False)
            
            # *peripheral_distractor* updates
            
            # if peripheral_distractor is starting this frame...
            if peripheral_distractor.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                peripheral_distractor.frameNStart = frameN  # exact frame index
                peripheral_distractor.tStart = t  # local t and not account for scr refresh
                peripheral_distractor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(peripheral_distractor, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'peripheral_distractor.started')
                # update status
                peripheral_distractor.status = STARTED
                peripheral_distractor.setAutoDraw(True)
            
            # if peripheral_distractor is active this frame...
            if peripheral_distractor.status == STARTED:
                # update params
                pass
            
            # if peripheral_distractor is stopping this frame...
            if peripheral_distractor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > peripheral_distractor.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    peripheral_distractor.tStop = t  # not accounting for scr refresh
                    peripheral_distractor.tStopRefresh = tThisFlipGlobal  # on global time
                    peripheral_distractor.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'peripheral_distractor.stopped')
                    # update status
                    peripheral_distractor.status = FINISHED
                    peripheral_distractor.setAutoDraw(False)
            
            # *pilot_response* updates
            waitOnFlip = False
            
            # if pilot_response is starting this frame...
            if pilot_response.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                pilot_response.frameNStart = frameN  # exact frame index
                pilot_response.tStart = t  # local t and not account for scr refresh
                pilot_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pilot_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pilot_response.started')
                # update status
                pilot_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(pilot_response.clock.reset)  # t=0 on next screen flip
            
            # if pilot_response is stopping this frame...
            if pilot_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > pilot_response.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    pilot_response.tStop = t  # not accounting for scr refresh
                    pilot_response.tStopRefresh = tThisFlipGlobal  # on global time
                    pilot_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pilot_response.stopped')
                    # update status
                    pilot_response.status = FINISHED
                    pilot_response.status = FINISHED
            if pilot_response.status == STARTED and not waitOnFlip:
                theseKeys = pilot_response.getKeys(keyList=['0','2'], ignoreKeys=["escape"], waitRelease=False)
                _pilot_response_allKeys.extend(theseKeys)
                if len(_pilot_response_allKeys):
                    pilot_response.keys = _pilot_response_allKeys[-1].name  # just the last key pressed
                    pilot_response.rt = _pilot_response_allKeys[-1].rt
                    pilot_response.duration = _pilot_response_allKeys[-1].duration
                    # was this correct?
                    if (pilot_response.keys == str('')) or (pilot_response.keys == ''):
                        pilot_response.corr = 1
                    else:
                        pilot_response.corr = 0
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pilot,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pilot.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pilot.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pilot.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pilot" ---
        for thisComponent in pilot.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pilot
        pilot.tStop = globalClock.getTime(format='float')
        pilot.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pilot.stopped', pilot.tStop)
        # Run 'End Routine' code from pilot_code
        if pilot_response.keys == str(int(pilot_correct_resp)):
            pilot_corr = 1
        else:
            pilot_corr = 0
        
        thisExp.addData('pilot_correct', pilot_corr)
        # check responses
        if pilot_response.keys in ['', [], None]:  # No response was made
            pilot_response.keys = None
            # was no response the correct answer?!
            if str('').lower() == 'none':
               pilot_response.corr = 1;  # correct non-response
            else:
               pilot_response.corr = 0;  # failed to respond (incorrectly)
        # store data for pilot_loop (TrialHandler)
        pilot_loop.addData('pilot_response.keys',pilot_response.keys)
        pilot_loop.addData('pilot_response.corr', pilot_response.corr)
        if pilot_response.keys != None:  # we had a response
            pilot_loop.addData('pilot_response.rt', pilot_response.rt)
            pilot_loop.addData('pilot_response.duration', pilot_response.duration)
        # Run 'End Routine' code from pilot_data
        thisExp.addData('phase', 'pilot')
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pilot.maxDurationReached:
            routineTimer.addTime(-pilot.maxDuration)
        elif pilot.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        # mark thisPilot_loop as finished
        if hasattr(thisPilot_loop, 'status'):
            thisPilot_loop.status = FINISHED
        # if awaiting a pause, pause now
        if pilot_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pilot_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'pilot_loop'
    pilot_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "main_instructions" ---
    # create an object to store info about Routine main_instructions
    main_instructions = data.Routine(
        name='main_instructions',
        components=[instructions_main, start_experiment],
    )
    main_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_experiment
    start_experiment.keys = []
    start_experiment.rt = []
    _start_experiment_allKeys = []
    # store start times for main_instructions
    main_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    main_instructions.tStart = globalClock.getTime(format='float')
    main_instructions.status = STARTED
    thisExp.addData('main_instructions.started', main_instructions.tStart)
    main_instructions.maxDuration = None
    # keep track of which components have finished
    main_instructionsComponents = main_instructions.components
    for thisComponent in main_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "main_instructions" ---
    thisExp.currentRoutine = main_instructions
    main_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_main* updates
        
        # if instructions_main is starting this frame...
        if instructions_main.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_main.frameNStart = frameN  # exact frame index
            instructions_main.tStart = t  # local t and not account for scr refresh
            instructions_main.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_main, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_main.started')
            # update status
            instructions_main.status = STARTED
            instructions_main.setAutoDraw(True)
        
        # if instructions_main is active this frame...
        if instructions_main.status == STARTED:
            # update params
            pass
        
        # *start_experiment* updates
        waitOnFlip = False
        
        # if start_experiment is starting this frame...
        if start_experiment.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_experiment.frameNStart = frameN  # exact frame index
            start_experiment.tStart = t  # local t and not account for scr refresh
            start_experiment.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_experiment, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_experiment.started')
            # update status
            start_experiment.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_experiment.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_experiment.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_experiment.status == STARTED and not waitOnFlip:
            theseKeys = start_experiment.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _start_experiment_allKeys.extend(theseKeys)
            if len(_start_experiment_allKeys):
                start_experiment.keys = _start_experiment_allKeys[-1].name  # just the last key pressed
                start_experiment.rt = _start_experiment_allKeys[-1].rt
                start_experiment.duration = _start_experiment_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=main_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            main_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if main_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in main_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "main_instructions" ---
    for thisComponent in main_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for main_instructions
    main_instructions.tStop = globalClock.getTime(format='float')
    main_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('main_instructions.stopped', main_instructions.tStop)
    # check responses
    if start_experiment.keys in ['', [], None]:  # No response was made
        start_experiment.keys = None
    thisExp.addData('start_experiment.keys',start_experiment.keys)
    if start_experiment.keys != None:  # we had a response
        thisExp.addData('start_experiment.rt', start_experiment.rt)
        thisExp.addData('start_experiment.duration', start_experiment.duration)
    thisExp.nextEntry()
    # the Routine "main_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    main_loop_1 = data.TrialHandler2(
        name='main_loop_1',
        nReps=5, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('C:/Users/shank/OneDrive/Desktop/Lavie_1995_Main_18_Conditions.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(main_loop_1)  # add the loop to the experiment
    thisMain_loop_1 = main_loop_1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMain_loop_1.rgb)
    if thisMain_loop_1 != None:
        for paramName in thisMain_loop_1:
            globals()[paramName] = thisMain_loop_1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMain_loop_1 in main_loop_1:
        main_loop_1.status = STARTED
        if hasattr(thisMain_loop_1, 'status'):
            thisMain_loop_1.status = STARTED
        currentLoop = main_loop_1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMain_loop_1.rgb)
        if thisMain_loop_1 != None:
            for paramName in thisMain_loop_1:
                globals()[paramName] = thisMain_loop_1[paramName]
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[fixation, letter1, letter2, letter3, letter4, letter5, letter6, peripheralDistractor, main_response],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from main_code
        if distractor_pos == "above":
            distractor_position = (0, 0.25)
        else:
            distractor_position = (0, -0.25)
        letter1.setText(pos1)
        letter2.setText(pos2)
        letter3.setText(pos3)
        letter4.setText(pos4
        )
        letter5.setText(pos5)
        letter6.setText(pos6)
        peripheralDistractor.setPos(distractor_position)
        peripheralDistractor.setText(distractor)
        # create starting attributes for main_response
        main_response.keys = []
        main_response.rt = []
        _main_response_allKeys = []
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        thisExp.currentRoutine = trial
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # if trial has changed, end Routine now
            if hasattr(thisMain_loop_1, 'status') and thisMain_loop_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *letter1* updates
            
            # if letter1 is starting this frame...
            if letter1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter1.frameNStart = frameN  # exact frame index
                letter1.tStart = t  # local t and not account for scr refresh
                letter1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter1.started')
                # update status
                letter1.status = STARTED
                letter1.setAutoDraw(True)
            
            # if letter1 is active this frame...
            if letter1.status == STARTED:
                # update params
                pass
            
            # if letter1 is stopping this frame...
            if letter1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter1.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter1.tStop = t  # not accounting for scr refresh
                    letter1.tStopRefresh = tThisFlipGlobal  # on global time
                    letter1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter1.stopped')
                    # update status
                    letter1.status = FINISHED
                    letter1.setAutoDraw(False)
            
            # *letter2* updates
            
            # if letter2 is starting this frame...
            if letter2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter2.frameNStart = frameN  # exact frame index
                letter2.tStart = t  # local t and not account for scr refresh
                letter2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter2.started')
                # update status
                letter2.status = STARTED
                letter2.setAutoDraw(True)
            
            # if letter2 is active this frame...
            if letter2.status == STARTED:
                # update params
                pass
            
            # if letter2 is stopping this frame...
            if letter2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter2.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter2.tStop = t  # not accounting for scr refresh
                    letter2.tStopRefresh = tThisFlipGlobal  # on global time
                    letter2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter2.stopped')
                    # update status
                    letter2.status = FINISHED
                    letter2.setAutoDraw(False)
            
            # *letter3* updates
            
            # if letter3 is starting this frame...
            if letter3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter3.frameNStart = frameN  # exact frame index
                letter3.tStart = t  # local t and not account for scr refresh
                letter3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter3.started')
                # update status
                letter3.status = STARTED
                letter3.setAutoDraw(True)
            
            # if letter3 is active this frame...
            if letter3.status == STARTED:
                # update params
                pass
            
            # if letter3 is stopping this frame...
            if letter3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter3.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter3.tStop = t  # not accounting for scr refresh
                    letter3.tStopRefresh = tThisFlipGlobal  # on global time
                    letter3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter3.stopped')
                    # update status
                    letter3.status = FINISHED
                    letter3.setAutoDraw(False)
            
            # *letter4* updates
            
            # if letter4 is starting this frame...
            if letter4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter4.frameNStart = frameN  # exact frame index
                letter4.tStart = t  # local t and not account for scr refresh
                letter4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter4.started')
                # update status
                letter4.status = STARTED
                letter4.setAutoDraw(True)
            
            # if letter4 is active this frame...
            if letter4.status == STARTED:
                # update params
                pass
            
            # if letter4 is stopping this frame...
            if letter4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter4.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter4.tStop = t  # not accounting for scr refresh
                    letter4.tStopRefresh = tThisFlipGlobal  # on global time
                    letter4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter4.stopped')
                    # update status
                    letter4.status = FINISHED
                    letter4.setAutoDraw(False)
            
            # *letter5* updates
            
            # if letter5 is starting this frame...
            if letter5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter5.frameNStart = frameN  # exact frame index
                letter5.tStart = t  # local t and not account for scr refresh
                letter5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter5.started')
                # update status
                letter5.status = STARTED
                letter5.setAutoDraw(True)
            
            # if letter5 is active this frame...
            if letter5.status == STARTED:
                # update params
                pass
            
            # if letter5 is stopping this frame...
            if letter5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter5.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter5.tStop = t  # not accounting for scr refresh
                    letter5.tStopRefresh = tThisFlipGlobal  # on global time
                    letter5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter5.stopped')
                    # update status
                    letter5.status = FINISHED
                    letter5.setAutoDraw(False)
            
            # *letter6* updates
            
            # if letter6 is starting this frame...
            if letter6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter6.frameNStart = frameN  # exact frame index
                letter6.tStart = t  # local t and not account for scr refresh
                letter6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter6.started')
                # update status
                letter6.status = STARTED
                letter6.setAutoDraw(True)
            
            # if letter6 is active this frame...
            if letter6.status == STARTED:
                # update params
                pass
            
            # if letter6 is stopping this frame...
            if letter6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter6.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter6.tStop = t  # not accounting for scr refresh
                    letter6.tStopRefresh = tThisFlipGlobal  # on global time
                    letter6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter6.stopped')
                    # update status
                    letter6.status = FINISHED
                    letter6.setAutoDraw(False)
            
            # *peripheralDistractor* updates
            
            # if peripheralDistractor is starting this frame...
            if peripheralDistractor.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                peripheralDistractor.frameNStart = frameN  # exact frame index
                peripheralDistractor.tStart = t  # local t and not account for scr refresh
                peripheralDistractor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(peripheralDistractor, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'peripheralDistractor.started')
                # update status
                peripheralDistractor.status = STARTED
                peripheralDistractor.setAutoDraw(True)
            
            # if peripheralDistractor is active this frame...
            if peripheralDistractor.status == STARTED:
                # update params
                pass
            
            # if peripheralDistractor is stopping this frame...
            if peripheralDistractor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > peripheralDistractor.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    peripheralDistractor.tStop = t  # not accounting for scr refresh
                    peripheralDistractor.tStopRefresh = tThisFlipGlobal  # on global time
                    peripheralDistractor.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'peripheralDistractor.stopped')
                    # update status
                    peripheralDistractor.status = FINISHED
                    peripheralDistractor.setAutoDraw(False)
            
            # *main_response* updates
            waitOnFlip = False
            
            # if main_response is starting this frame...
            if main_response.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                main_response.frameNStart = frameN  # exact frame index
                main_response.tStart = t  # local t and not account for scr refresh
                main_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(main_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'main_response.started')
                # update status
                main_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(main_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(main_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if main_response is stopping this frame...
            if main_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > main_response.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    main_response.tStop = t  # not accounting for scr refresh
                    main_response.tStopRefresh = tThisFlipGlobal  # on global time
                    main_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'main_response.stopped')
                    # update status
                    main_response.status = FINISHED
                    main_response.status = FINISHED
            if main_response.status == STARTED and not waitOnFlip:
                theseKeys = main_response.getKeys(keyList=['0', '2'], ignoreKeys=["escape"], waitRelease=False)
                _main_response_allKeys.extend(theseKeys)
                if len(_main_response_allKeys):
                    main_response.keys = _main_response_allKeys[-1].name  # just the last key pressed
                    main_response.rt = _main_response_allKeys[-1].rt
                    main_response.duration = _main_response_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # Run 'End Routine' code from main_code
        if main_response.keys == str(int(correct_resp)):
            main_corr = 1
        else:
            main_corr = 0
        
        thisExp.addData('main_correct', main_corr)
        
        # check responses
        if main_response.keys in ['', [], None]:  # No response was made
            main_response.keys = None
        main_loop_1.addData('main_response.keys',main_response.keys)
        if main_response.keys != None:  # we had a response
            main_loop_1.addData('main_response.rt', main_response.rt)
            main_loop_1.addData('main_response.duration', main_response.duration)
        # Run 'End Routine' code from main_data
        thisExp.addData('phase', 'main')
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial.maxDurationReached:
            routineTimer.addTime(-trial.maxDuration)
        elif trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        # mark thisMain_loop_1 as finished
        if hasattr(thisMain_loop_1, 'status'):
            thisMain_loop_1.status = FINISHED
        # if awaiting a pause, pause now
        if main_loop_1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            main_loop_1.status = STARTED
        thisExp.nextEntry()
        
    # completed 5 repeats of 'main_loop_1'
    main_loop_1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "break_3min" ---
    # create an object to store info about Routine break_3min
    break_3min = data.Routine(
        name='break_3min',
        components=[break_instructions, break_text],
    )
    break_3min.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for break_text
    break_text.keys = []
    break_text.rt = []
    _break_text_allKeys = []
    # store start times for break_3min
    break_3min.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_3min.tStart = globalClock.getTime(format='float')
    break_3min.status = STARTED
    thisExp.addData('break_3min.started', break_3min.tStart)
    break_3min.maxDuration = None
    # keep track of which components have finished
    break_3minComponents = break_3min.components
    for thisComponent in break_3min.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_3min" ---
    thisExp.currentRoutine = break_3min
    break_3min.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 180.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *break_instructions* updates
        
        # if break_instructions is starting this frame...
        if break_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            break_instructions.frameNStart = frameN  # exact frame index
            break_instructions.tStart = t  # local t and not account for scr refresh
            break_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(break_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'break_instructions.started')
            # update status
            break_instructions.status = STARTED
            break_instructions.setAutoDraw(True)
        
        # if break_instructions is active this frame...
        if break_instructions.status == STARTED:
            # update params
            pass
        
        # if break_instructions is stopping this frame...
        if break_instructions.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > break_instructions.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                break_instructions.tStop = t  # not accounting for scr refresh
                break_instructions.tStopRefresh = tThisFlipGlobal  # on global time
                break_instructions.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_instructions.stopped')
                # update status
                break_instructions.status = FINISHED
                break_instructions.setAutoDraw(False)
        
        # *break_text* updates
        waitOnFlip = False
        
        # if break_text is starting this frame...
        if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            break_text.frameNStart = frameN  # exact frame index
            break_text.tStart = t  # local t and not account for scr refresh
            break_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'break_text.started')
            # update status
            break_text.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(break_text.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(break_text.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if break_text is stopping this frame...
        if break_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > break_text.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                break_text.tStop = t  # not accounting for scr refresh
                break_text.tStopRefresh = tThisFlipGlobal  # on global time
                break_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.stopped')
                # update status
                break_text.status = FINISHED
                break_text.status = FINISHED
        if break_text.status == STARTED and not waitOnFlip:
            theseKeys = break_text.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _break_text_allKeys.extend(theseKeys)
            if len(_break_text_allKeys):
                break_text.keys = _break_text_allKeys[-1].name  # just the last key pressed
                break_text.rt = _break_text_allKeys[-1].rt
                break_text.duration = _break_text_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=break_3min,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            break_3min.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if break_3min.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in break_3min.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_3min" ---
    for thisComponent in break_3min.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_3min
    break_3min.tStop = globalClock.getTime(format='float')
    break_3min.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_3min.stopped', break_3min.tStop)
    # check responses
    if break_text.keys in ['', [], None]:  # No response was made
        break_text.keys = None
    thisExp.addData('break_text.keys',break_text.keys)
    if break_text.keys != None:  # we had a response
        thisExp.addData('break_text.rt', break_text.rt)
        thisExp.addData('break_text.duration', break_text.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_3min.maxDurationReached:
        routineTimer.addTime(-break_3min.maxDuration)
    elif break_3min.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-180.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    main_loop_2 = data.TrialHandler2(
        name='main_loop_2',
        nReps=5, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('C:/Users/shank/OneDrive/Desktop/Lavie_1995_Main_18_Conditions.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(main_loop_2)  # add the loop to the experiment
    thisMain_loop_2 = main_loop_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMain_loop_2.rgb)
    if thisMain_loop_2 != None:
        for paramName in thisMain_loop_2:
            globals()[paramName] = thisMain_loop_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMain_loop_2 in main_loop_2:
        main_loop_2.status = STARTED
        if hasattr(thisMain_loop_2, 'status'):
            thisMain_loop_2.status = STARTED
        currentLoop = main_loop_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMain_loop_2.rgb)
        if thisMain_loop_2 != None:
            for paramName in thisMain_loop_2:
                globals()[paramName] = thisMain_loop_2[paramName]
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[fixation, letter1, letter2, letter3, letter4, letter5, letter6, peripheralDistractor, main_response],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from main_code
        if distractor_pos == "above":
            distractor_position = (0, 0.25)
        else:
            distractor_position = (0, -0.25)
        letter1.setText(pos1)
        letter2.setText(pos2)
        letter3.setText(pos3)
        letter4.setText(pos4
        )
        letter5.setText(pos5)
        letter6.setText(pos6)
        peripheralDistractor.setPos(distractor_position)
        peripheralDistractor.setText(distractor)
        # create starting attributes for main_response
        main_response.keys = []
        main_response.rt = []
        _main_response_allKeys = []
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        thisExp.currentRoutine = trial
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # if trial has changed, end Routine now
            if hasattr(thisMain_loop_2, 'status') and thisMain_loop_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *letter1* updates
            
            # if letter1 is starting this frame...
            if letter1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter1.frameNStart = frameN  # exact frame index
                letter1.tStart = t  # local t and not account for scr refresh
                letter1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter1.started')
                # update status
                letter1.status = STARTED
                letter1.setAutoDraw(True)
            
            # if letter1 is active this frame...
            if letter1.status == STARTED:
                # update params
                pass
            
            # if letter1 is stopping this frame...
            if letter1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter1.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter1.tStop = t  # not accounting for scr refresh
                    letter1.tStopRefresh = tThisFlipGlobal  # on global time
                    letter1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter1.stopped')
                    # update status
                    letter1.status = FINISHED
                    letter1.setAutoDraw(False)
            
            # *letter2* updates
            
            # if letter2 is starting this frame...
            if letter2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter2.frameNStart = frameN  # exact frame index
                letter2.tStart = t  # local t and not account for scr refresh
                letter2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter2.started')
                # update status
                letter2.status = STARTED
                letter2.setAutoDraw(True)
            
            # if letter2 is active this frame...
            if letter2.status == STARTED:
                # update params
                pass
            
            # if letter2 is stopping this frame...
            if letter2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter2.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter2.tStop = t  # not accounting for scr refresh
                    letter2.tStopRefresh = tThisFlipGlobal  # on global time
                    letter2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter2.stopped')
                    # update status
                    letter2.status = FINISHED
                    letter2.setAutoDraw(False)
            
            # *letter3* updates
            
            # if letter3 is starting this frame...
            if letter3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter3.frameNStart = frameN  # exact frame index
                letter3.tStart = t  # local t and not account for scr refresh
                letter3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter3.started')
                # update status
                letter3.status = STARTED
                letter3.setAutoDraw(True)
            
            # if letter3 is active this frame...
            if letter3.status == STARTED:
                # update params
                pass
            
            # if letter3 is stopping this frame...
            if letter3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter3.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter3.tStop = t  # not accounting for scr refresh
                    letter3.tStopRefresh = tThisFlipGlobal  # on global time
                    letter3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter3.stopped')
                    # update status
                    letter3.status = FINISHED
                    letter3.setAutoDraw(False)
            
            # *letter4* updates
            
            # if letter4 is starting this frame...
            if letter4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter4.frameNStart = frameN  # exact frame index
                letter4.tStart = t  # local t and not account for scr refresh
                letter4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter4.started')
                # update status
                letter4.status = STARTED
                letter4.setAutoDraw(True)
            
            # if letter4 is active this frame...
            if letter4.status == STARTED:
                # update params
                pass
            
            # if letter4 is stopping this frame...
            if letter4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter4.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter4.tStop = t  # not accounting for scr refresh
                    letter4.tStopRefresh = tThisFlipGlobal  # on global time
                    letter4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter4.stopped')
                    # update status
                    letter4.status = FINISHED
                    letter4.setAutoDraw(False)
            
            # *letter5* updates
            
            # if letter5 is starting this frame...
            if letter5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter5.frameNStart = frameN  # exact frame index
                letter5.tStart = t  # local t and not account for scr refresh
                letter5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter5.started')
                # update status
                letter5.status = STARTED
                letter5.setAutoDraw(True)
            
            # if letter5 is active this frame...
            if letter5.status == STARTED:
                # update params
                pass
            
            # if letter5 is stopping this frame...
            if letter5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter5.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter5.tStop = t  # not accounting for scr refresh
                    letter5.tStopRefresh = tThisFlipGlobal  # on global time
                    letter5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter5.stopped')
                    # update status
                    letter5.status = FINISHED
                    letter5.setAutoDraw(False)
            
            # *letter6* updates
            
            # if letter6 is starting this frame...
            if letter6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                letter6.frameNStart = frameN  # exact frame index
                letter6.tStart = t  # local t and not account for scr refresh
                letter6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(letter6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'letter6.started')
                # update status
                letter6.status = STARTED
                letter6.setAutoDraw(True)
            
            # if letter6 is active this frame...
            if letter6.status == STARTED:
                # update params
                pass
            
            # if letter6 is stopping this frame...
            if letter6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > letter6.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    letter6.tStop = t  # not accounting for scr refresh
                    letter6.tStopRefresh = tThisFlipGlobal  # on global time
                    letter6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'letter6.stopped')
                    # update status
                    letter6.status = FINISHED
                    letter6.setAutoDraw(False)
            
            # *peripheralDistractor* updates
            
            # if peripheralDistractor is starting this frame...
            if peripheralDistractor.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                peripheralDistractor.frameNStart = frameN  # exact frame index
                peripheralDistractor.tStart = t  # local t and not account for scr refresh
                peripheralDistractor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(peripheralDistractor, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'peripheralDistractor.started')
                # update status
                peripheralDistractor.status = STARTED
                peripheralDistractor.setAutoDraw(True)
            
            # if peripheralDistractor is active this frame...
            if peripheralDistractor.status == STARTED:
                # update params
                pass
            
            # if peripheralDistractor is stopping this frame...
            if peripheralDistractor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > peripheralDistractor.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    peripheralDistractor.tStop = t  # not accounting for scr refresh
                    peripheralDistractor.tStopRefresh = tThisFlipGlobal  # on global time
                    peripheralDistractor.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'peripheralDistractor.stopped')
                    # update status
                    peripheralDistractor.status = FINISHED
                    peripheralDistractor.setAutoDraw(False)
            
            # *main_response* updates
            waitOnFlip = False
            
            # if main_response is starting this frame...
            if main_response.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                main_response.frameNStart = frameN  # exact frame index
                main_response.tStart = t  # local t and not account for scr refresh
                main_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(main_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'main_response.started')
                # update status
                main_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(main_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(main_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if main_response is stopping this frame...
            if main_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > main_response.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    main_response.tStop = t  # not accounting for scr refresh
                    main_response.tStopRefresh = tThisFlipGlobal  # on global time
                    main_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'main_response.stopped')
                    # update status
                    main_response.status = FINISHED
                    main_response.status = FINISHED
            if main_response.status == STARTED and not waitOnFlip:
                theseKeys = main_response.getKeys(keyList=['0', '2'], ignoreKeys=["escape"], waitRelease=False)
                _main_response_allKeys.extend(theseKeys)
                if len(_main_response_allKeys):
                    main_response.keys = _main_response_allKeys[-1].name  # just the last key pressed
                    main_response.rt = _main_response_allKeys[-1].rt
                    main_response.duration = _main_response_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # Run 'End Routine' code from main_code
        if main_response.keys == str(int(correct_resp)):
            main_corr = 1
        else:
            main_corr = 0
        
        thisExp.addData('main_correct', main_corr)
        
        # check responses
        if main_response.keys in ['', [], None]:  # No response was made
            main_response.keys = None
        main_loop_2.addData('main_response.keys',main_response.keys)
        if main_response.keys != None:  # we had a response
            main_loop_2.addData('main_response.rt', main_response.rt)
            main_loop_2.addData('main_response.duration', main_response.duration)
        # Run 'End Routine' code from main_data
        thisExp.addData('phase', 'main')
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial.maxDurationReached:
            routineTimer.addTime(-trial.maxDuration)
        elif trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        # mark thisMain_loop_2 as finished
        if hasattr(thisMain_loop_2, 'status'):
            thisMain_loop_2.status = FINISHED
        # if awaiting a pause, pause now
        if main_loop_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            main_loop_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 5 repeats of 'main_loop_2'
    main_loop_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thank_you" ---
    # create an object to store info about Routine thank_you
    thank_you = data.Routine(
        name='thank_you',
        components=[text_2, key_resp_2],
    )
    thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for thank_you
    thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thank_you.tStart = globalClock.getTime(format='float')
    thank_you.status = STARTED
    thisExp.addData('thank_you.started', thank_you.tStart)
    thank_you.maxDuration = None
    # keep track of which components have finished
    thank_youComponents = thank_you.components
    for thisComponent in thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    thisExp.currentRoutine = thank_you
    thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thank_you,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            thank_you.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if thank_you.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thank_you
    thank_you.tStop = globalClock.getTime(format='float')
    thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thank_you.stopped', thank_you.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "thank_you" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

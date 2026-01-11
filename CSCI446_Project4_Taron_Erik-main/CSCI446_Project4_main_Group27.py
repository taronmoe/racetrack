import math
import numpy as np
import os
import random 
import sys
import re
import copy as cp
from Track import Track
import matplotlib.pyplot as plt

def fileImport(fileName): # brings file into program as str numpy array
    inputText = ''
    with open(fileName, "r") as f:
        inputText = f.read()
    print(inputText)
    textRows = inputText.split('\n')
    dimTxt = textRows[0].split(',')

    numRows = int(dimTxt[0])
    numCols = int(dimTxt[1])

    inputTextArray = np.zeros((numRows, numCols), dtype = str)

    for row in range(numRows):
        # currentRow = textRows[row + 1].split()
        for col in range(numCols):
            inputTextArray[row][col] = textRows[row + 1][col]
            
    # print(inputTextArray)
    
    return inputTextArray

def createTrack(inputTextArray, CRASH_POS):
    rows = inputTextArray.shape[0]
    cols = inputTextArray.shape[1]

    trackIntegers = np.zeros((rows, cols), dtype = int)

    # 1 is 'S' aka Start, 2 is 'F' aka Finish, 0 is '.' aka track, 3 is '#', or wall. 
    for row in range(rows):
        for col in range(cols):
            character = inputTextArray[row][col]
            if (character == 'S'):
                trackIntegers[row][col] = 1
            if (character == 'F'):
                trackIntegers[row][col] = 2
            if (character == '#'):
                trackIntegers[row][col] = 3

    crashReset = False

    if (CRASH_POS == 'STRT'):
        crashReset = True
    elif (CRASH_POS != 'NRST'):
        print("CRASH_POS invalid, defaulting to NRST setting (nearest non-crash position)")
        
    track = Track(trackIntegers , crashReset, inputTextArray)
    
    return track

import matplotlib.pyplot as plt   # ← only this import is needed

def saveOutput(GROUP_ID, ALGORITHM, TRACK_NAME, CRASH_POS, track):

    trackPathArr = TRACK_NAME.split('/')
    trackName = trackPathArr[-1]
    # txtFileName = GROUP_ID + "*" + ALGORITHM + "*" + trackName[:-4] + "*" + CRASH_POS + ".txt"
    
    writeArray = track.getInputTextArray().copy() 
    path = track.getBestPath()
    writeString = ''
    
    for pos in path:
        writeArray[pos[0]][pos[1]] = 'P'
    
    for row in writeArray:
        for col in row:
            writeString += col
        writeString += '\n'
    
    #with open(txtFileName, "w") as f:
        #f.write(writeString)
    
    print("Best Path Taken: ")
    print(writeString)

    pngFileName = GROUP_ID + "_" + ALGORITHM + "_" + trackName[:-4] + "_" + CRASH_POS + ".png"
    
    grid = track.getInputTextArray()
    rows, cols = grid.shape

    fig = plt.figure(figsize=(max(8, cols/10), max(8, rows/10)), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    # Draw tiles
    for r in range(rows):
        for c in range(cols):
            char = grid[r, c]
            color = {'#': 'gray', 'F': 'limegreen', 'S': 'royalblue'}.get(char, 'white')
            ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                       facecolor=color, edgecolor='black', linewidth=0.5))

            # Small yellow square only on original '.' cells that are on the path
            if (r, c) in path and grid[r, c] == '.':
                ax.add_patch(plt.Rectangle((c + 0.25, r + 0.25), 0.5, 0.5,
                                           facecolor='yellow', edgecolor='orange', linewidth=1))

    # Gold line connecting path centers
    if len(path) > 1:
        x = [p[1] + 0.5 for p in path]
        y = [p[0] + 0.5 for p in path]
        ax.plot(x, y, color='gold', linewidth=6, solid_capstyle='round')

    # Dark circles on Start and Finish
    if path:
        sr, sc = path[0]
        fr, fc = path[-1]
        ax.add_patch(plt.Circle((sc + 0.5, sr + 0.5), 0.4, color='darkblue'))
        ax.add_patch(plt.Circle((fc + 0.5, fr + 0.5), 0.4, color='darkgreen'))

    ax.set_title(f"{ALGORITHM} – {track.getBestMoves()} moves", fontsize=14, pad=20)
    plt.savefig(pngFileName, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

def main(GROUP_ID, ALGORITHM, TRACK_NAME, CRASH_POS): 

    inputTextArray = fileImport(TRACK_NAME)
    track = createTrack(inputTextArray, CRASH_POS)
    
    if (ALGORITHM == 'ValItr'):
        # code to run variable elimination
        # will be method on Network class
        print("run Value Iteration")
        track.doValueIteration()
        
    elif (ALGORITHM == 'QLrng'):
        # code to run gibbs sampling
        # will be method on Network class
        print("run Q-Learning")
        track.doQLearning()

    elif (ALGORITHM == 'SARSA'):
        # code to run gibbs sampling
        # will be method on Network class
        print("run State-Action-Reward-State-Action")
        track.doSARSA()
        
    else:
        print("Not a valid algorithm. Terminating...")
        sys.exit() # exit program

    print("Moves of Best Run: " + str(track.getBestMoves()))
    print("Operations to Find Solution: " + str(track.getOperations()))

    saveOutput(GROUP_ID, ALGORITHM, TRACK_NAME, CRASH_POS, track)
    
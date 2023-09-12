# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:04:44 2023

@author: rafiparker
"""
import pandas as pd
import numpy as np


class PreProcessIndividual:
    
    def __init__(self,pixelLength,frameRate,cutoffProb,relevantLabels,stimTimes,distanceThresh,mouse,stimPattern,smoothingFactor):
        
        self.pixelLength = pixelLength
        self.frameRate = frameRate
        self.cutoffProb = cutoffProb
        self.relevantLabels = relevantLabels
        self.stimTimes = stimTimes
        self.distanceThresh = distanceThresh
        self.mouse = mouse
        self.stimPattern = stimPattern
        self.smoothingFactor = smoothingFactor +1
        self.pairs = (
            [('Left','Before'),('Left','During')],
            [('Left','During'),('Left','After')],
            [('Left','Before'),('Left','After')],
            
            [('Right','Before'),('Right','During')],
            [('Right','During'),('Right','After')],
            [('Right','Before'),('Right','After')],
            
            [('Left','Before'),('Right','Before')],
            [('Left','During'),('Right','During')],
            [('Left','After'),('Right','After')])
            
#%%        
    def importDLC(self,path):
        DLCfile = pd.read_csv(path,skiprows=[0,2]).drop(['bodyparts','Fiber','Fiber.1','Fiber.2'],axis=1)
        for col in DLCfile.columns:
            if col[-1] == '2':
                DLCfile = DLCfile.rename(columns={col:col[:-2]+'_prob'})
            elif col[-1] == '1':
                DLCfile = DLCfile.rename(columns={col:col[:-2]+'_y'})
            else:
                DLCfile = DLCfile.rename(columns={col:col+'_x'})
        self.DLCfile = DLCfile
    
    def cleanOutliers(self):
        DLCfile=self.DLCfile
        relevantLabels = self.relevantLabels
        cutoffProb = self.cutoffProb
        
        allOutliers = []
        numOutliers = []
        
        for label in relevantLabels:
            prob = np.array(DLCfile[label+'_prob'])
            outliers = np.where(prob<cutoffProb)[0]
            allOutliers = allOutliers + list(outliers)
            numOutliers.append(np.round(len(outliers)/len(prob),4)*100)
            
        cleanedDLC = DLCfile.drop(allOutliers)
        
        self.cleanDLC = cleanedDLC
        self.allOutliers = allOutliers
        self.numOutliers = numOutliers
        
        return numOutliers
        
#%%    
    def distanceTraveled(self):
        cleanedDLC = self.cleanDLC
        pixelLength = self.pixelLength
        frameRate = self.frameRate
        
        delta = []
        for frame in range(len(cleanedDLC.index)-1):
            deltaX = cleanedDLC['Spine_x'].iloc[frame+1] - cleanedDLC['Spine_x'].iloc[frame]
            deltaY = cleanedDLC['Spine_y'].iloc[frame+1] - cleanedDLC['Spine_y'].iloc[frame]
            delta.append( np.sqrt( np.power(deltaX,2) + np.power(deltaY,2))/(frame+1-frame))
        
        deltaCm = np.array(delta)*pixelLength
        outliersDist = np.where(deltaCm > (np.mean(deltaCm)+(8*np.std(deltaCm))))[0]
        deltaCm = np.delete(deltaCm,outliersDist)
        deltaCm = np.append(deltaCm,deltaCm[-1])
        speedSec = deltaCm*frameRate
        
        self.distance = deltaCm
        self.speed = speedSec
        
        removed = [self.cleanDLC.index[i] for i in outliersDist]
        self.cleanDLC = self.cleanDLC.drop(removed)
        self.allOutliers = self.allOutliers +removed
        
        return deltaCm, speedSec
    #%%
    def turnAngle(self):
        cleanedDLC = self.cleanDLC
        distanceCm = self.distance
        relevantLabels = self.relevantLabels
        distanceThresh = self.distanceThresh
    
        bodyparts = []
        for label in relevantLabels:
            bodyparts.append([cleanedDLC[label+'_x'],cleanedDLC[label+'_y']])
            
        left = [cleanedDLC['LS_x'],cleanedDLC['LS_y']]
        right = [cleanedDLC['RS_x'],cleanedDLC['RS_y']]
            
        theta,midPoint = PreProcessIndividual.getAngle(bodyparts[0],bodyparts[1],bodyparts[2])
        turnLeft,directionTurn = PreProcessIndividual.direction(midPoint,left,right)
        
        
        onlyMove = np.where(np.array(distanceCm) > distanceThresh)[0]
        thetaMove = np.array(theta)[onlyMove]
        if min(thetaMove)<90:
            minAngle = 90
        else:
            minAngle = min(thetaMove)
        sharpAngle = 170 - (170-minAngle)*0.65 #sharpest 35%
        mediumAngle = 170 - (170-minAngle)*0.25 #medium 40%
        
        angles = {'Sharp':sharpAngle,'Medium':mediumAngle,'Broad':170,'Minimum':minAngle}
        
        angleJudgements = [PreProcessIndividual.qualifyAngle(theta[i],turnLeft[i],\
                           distanceCm[i],sharpAngle,mediumAngle,distanceThresh)\
                           for i in range(len(theta))]
        
        angleJudgements = PreProcessIndividual.smoothAngle(angleJudgements,self.smoothingFactor,'Left')
        angleJudgements = PreProcessIndividual.smoothAngle(angleJudgements,self.smoothingFactor,'Right')
        
        self.theta = theta
        self.midPoint = midPoint
        self.direction = directionTurn
        self.angleJudgements = angleJudgements
        self.angleCategories = angles
        
        return angleJudgements
            
    def getAngle(bodypart1:list,bodypart2:list,bodypart3:list):
        v1 = np.subtract(bodypart3,bodypart2)
        v2 = np.subtract(bodypart1,bodypart2)
        
        theta = []
        for frame in range(len(v1[0])):
            dot = np.dot(v2[:,frame],v1[:,frame])
            norm = np.linalg.norm(v2[:,frame])*np.linalg.norm(v1[:,frame])
            cosTheta = dot/norm
            theta.append(np.rad2deg(np.arccos(cosTheta)))
            
        midPoint = np.divide( np.add(bodypart1,bodypart3),2)
        return theta, midPoint
    
    def direction(midPoint,left:list,right:list):
        diffLeft = np.subtract(midPoint,np.array(left))
        diffRight = np.subtract(midPoint,np.array(right))
        
        distLeft = np.sqrt(np.add(np.power(diffLeft[0],2),np.power(diffLeft[1],2)))
        distRight = np.sqrt(np.add(np.power(diffRight[0],2),np.power(diffRight[1],2)))
        
        turnLeft = (distLeft<distRight)
        
        turnRight = (distRight<distLeft)
        
        directionTurn = ['Left' if i else 'Straight' for i in turnLeft]
        for i in range(len(turnRight)):
            if turnRight[i]:
                directionTurn[i] = 'Right'
        for i in range(len(directionTurn)-5):
            directionTurn
        
        return turnLeft, directionTurn
    
    def qualifyAngle(theta,Left,distance,sharpAngle,mediumAngle,distanceThresh):
        if Left:
            direction = 'Left_'
        else:
            direction = 'Right_'
        if theta>170 or distance<distanceThresh:
            return 'Straight'
        elif theta > mediumAngle:
            return direction+'Broad'
        elif theta > sharpAngle:
            return direction+'Medium'
        else:
            return direction+'Sharp'
            
    def smoothAngle(angleJudgements,smoothingFactor,direction):
        angleJudgements = np.array(angleJudgements)
        for judge in range(len(angleJudgements)-(smoothingFactor+1)):
            if angleJudgements[judge][:len(direction)] == direction and angleJudgements[judge+1]=='Straight':
                nextFew = angleJudgements[judge+1:judge+smoothingFactor]
                nextFew = [i[:len(direction)] for i in nextFew]
                if direction in nextFew:
                    nextOne = np.where(np.array(nextFew)==direction)[0][-1]
                    angleJudgements[judge+1:judge+nextOne+1] = angleJudgements[judge]
                    judge+=nextOne
        return angleJudgements.tolist()
        
#%%
    def alignOutliers(self):
        index = self.cleanDLC.index
        stimTimes = self.stimTimes
        shiftTimes = []
        for time in stimTimes:
            diff = np.array(index)-time
            shiftTimes.append(np.where(abs(diff)==min(abs(diff)))[0][0])
        self.shiftTimes = shiftTimes
        return shiftTimes
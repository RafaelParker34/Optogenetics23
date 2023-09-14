# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:45:09 2023

@author: rafiparker
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statannotations.Annotator import Annotator

import scipy.stats as stats
import numpy as np
import pandas as pd

#%%
def generateReport(preprocessed,savePath):
    plot1 = plotLikelihood(preprocessed)
    plot2 = plotDistance(preprocessed)
    plot3 = plotTurning(preprocessed)
    plot4 = plotPath(preprocessed)
    plot5 = timeTurning(preprocessed)
    plot6 = distTurning(preprocessed)
    plot7 = overallDist(preprocessed)
    plot8 = cumulativeAngle(preprocessed)

    pp = PdfPages(savePath)
    pp.savefig(plot1,bbox_inches='tight')
    pp.savefig(plot2,bbox_inches='tight')
    pp.savefig(plot3,bbox_inches='tight')
    pp.savefig(plot4,bbox_inches='tight')
    pp.savefig(plot5,bbox_inches='tight')
    pp.savefig(plot6,bbox_inches='tight')
    pp.savefig(plot7,bbox_inches='tight')
    pp.savefig(plot8,bbox_inches='tight')
    if preprocessed.mouse == '0510' or '0513':
        plot9 = duringStimulation(preprocessed)
        pp.savefig(plot9,bbox_inches='tight')

    pp.close()


#%%        
def bins30s(preprocessed,subCategory):
    
    ending = preprocessed.shiftTimes[2]
    
    outliers = preprocessed.allOutliers
    times = np.array(preprocessed.cleanDLC.index)
    bins = [0]
    i = 30*25
    while i < ending:
        while i in outliers: #Might happend that an exact 30sec mark is removed
            i+=1
        edge = np.where(times==(i))[0][0]
        bins.append(edge)
        i+=30*25
    bins.append(ending)

    turns = []
    category = []
    start = preprocessed.shiftTimes[0]
    finish = preprocessed.shiftTimes[1]
    
    for i in range(len(bins)-1):
        binSingle = pd.Series(subCategory[bins[i]:bins[i+1]])
        turns.append(binSingle)
        if np.abs(bins[i]- start)<125: #range in case the mark was removed
            category.append(i)
        elif np.abs(bins[i]- finish)<125:
            category.append(i)
    
    return turns, category
#%%
def plotLikelihood(preprocessed):
    fig,ax = plt.subplots()
    ax.set_title('Likelihood of Labels being Correct')
    ax.set_ylabel('Likelihood')
    ax.set_xlabel('Frame')
    ax.set_ylim(0,1)
    lines = []
    i = 1
    for label in preprocessed.relevantLabels:
        lines.append(ax.plot(preprocessed.DLCfile[label+'_prob'])[0])
        ax.text(1,i/15,label+' : '+str(preprocessed.numOutliers[i-1])+'%')
        i+=1
    ax.legend(lines,preprocessed.relevantLabels,loc='lower right')
    ax.axhline(y=0.7,color='k',linestyle='--')
    return fig
#%%    
def plotDistance(preprocessed):
    fig,ax = plt.subplots(3,1)
    plt.subplots_adjust(hspace=1)
    fig.set_figwidth(12)
    fig.set_figheight(9)
    ax[0].plot(np.cumsum(preprocessed.distance))
    ax[0].set_title('Distance Over All Frames (Spine)')
    ax[0].set_ylabel('Cummulative Distance (Cm)')
    
    
    N = len(preprocessed.distance)
    x = np.arange(N)
    segs = [[(x[i],0),(x[i+1],preprocessed.distance[i])] for i in range(len(x)-1)]
    line_segments = LineCollection(segs,linestyles='solid')
    
    
    ax[1].add_collection(line_segments)
    ax[1].set_title('Distance Traveled Per Frame (Spine)')
    ax[1].set_ylabel('Distance(Cm)')
    ax[1].set_ylim(0,max(preprocessed.distance))
    
    segs = [[(x[i],0),(x[i+1],preprocessed.speed[i])] for i in range(len(x)-1)]
    line_segments = LineCollection(segs,linestyles='solid')
    
    ax[2].add_collection(line_segments)
    ax[2].set_title('Speed Per Frame (Spine)')
    ax[2].set_ylabel('Speed (Cm/Sec)')
    ax[2].set_ylim(0,max(preprocessed.speed))
    
    for i in range(3):
        ax[i].set_xlabel('Frame')
        ax[i].axvline(x=preprocessed.shiftTimes[0],color='k',linestyle='--')
        ax[i].axvline(x=preprocessed.shiftTimes[1],color='k',linestyle='--')
    return fig
#%%       
def plotTurning(preprocessed):
    fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
    fig.set_figwidth(16)
    fig.set_figheight(8)
    ax[0].set_ylabel('# of Frames')
    colors = ['mediumslateblue','lightseagreen','coral']
    
    start = preprocessed.shiftTimes[0]
    finish = preprocessed.shiftTimes[1]
    length =min([start,finish-start,preprocessed.shiftTimes[2]-finish])
    duration = str(np.round(length/25/60).astype(int))
    
    fig.suptitle('Turn Angles During Movement Over '+duration+' Minutes')
    for i in range(3):
        turnFrames = np.where(np.array(preprocessed.angleJudgements[start-length:start]) != 'Straight')[0]
        turns = np.array(preprocessed.theta[start-length:start])[turnFrames]
        binning = np.linspace(90,170,16)
        
        if i ==0:
            ax[0].hist(turns,alpha=0.8,color=colors[i],bins=binning)
            ax[0].hist(turns,color=colors[i],histtype='step',bins=binning)
            start +=length
        elif i ==1:
            ax[0].hist(turns,alpha=0.6,color=colors[i],bins=binning)
            ax[0].hist(turns,color=colors[i],histtype='step',bins=binning)
            ax[1].hist(turns,alpha=0.8,color=colors[i],bins=binning)
            ax[1].hist(turns,color=colors[i],histtype='step',bins=binning)
            start +=length
        else:
            ax[1].hist(turns,alpha=0.6,color=colors[i],bins=binning)
            ax[1].hist(turns,color=colors[i],histtype='step',bins=binning)
   
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ['Before','During','After']
    maxY = ax[0].get_ylim()[1]
   
    for i in range(2):
        ax[i].legend(handles[i:i+2],labels[i:i+2],loc='upper left')
        ax[i].set_xlabel('Angle Measure (degrees)')
        ax[i].set_xticks([90,100,110,120,130,140,150,160,170])
        ax[i].axvline(x=preprocessed.angleCategories['Sharp'],color='k',linestyle='--')
        ax[i].axvline(x=preprocessed.angleCategories['Medium'],color='k',linestyle='--')
        ax[i].axvline(x=preprocessed.angleCategories['Broad'],color='k',linestyle='--')
        ax[i].text((90+preprocessed.angleCategories['Sharp'])/2.2,maxY-maxY*.7,'Sharp Angle')
        ax[i].text((preprocessed.angleCategories['Sharp']+preprocessed.angleCategories['Medium'])/2.15,maxY-maxY*.4,'Medium Angle')
        ax[i].text((preprocessed.angleCategories['Medium']+preprocessed.angleCategories['Broad'])/2.1,maxY-maxY*.1,'Broad Angle')
        
    return fig
#%%
def plotPath(preprocessed):
    fig, ax =  plt.subplots(1,3,sharex=True,sharey=True) 
    fig.set_figwidth(18)
    fig.set_figheight(6)
    
    start = preprocessed.shiftTimes[0]
    finish = preprocessed.shiftTimes[1]
    length = min([start,finish-start,preprocessed.shiftTimes[2]-finish])
    duration = str(np.round(length/25/60).astype(int))
    
    colorDict = {'Straight':'mediumblue','Right_Broad':'lightcoral','Right_Medium':'firebrick',\
                 'Right_Sharp':'maroon','Left_Broad':'mediumseagreen','Left_Medium':'forestgreen',\
                     'Left_Sharp':'darkgreen'}
    plt.suptitle('Path Over '+duration+' Minutes')
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in list(colorDict.values())]
    labels = ['St','RB','RM','RS','LB','LM','LS']
    for i in range(3):
        colorMap =[colorDict[preprocessed.angleJudgements[i]] for i in range(start-length,start)]
        pathX = list(preprocessed.cleanDLC['Spine_x'][start-length:start])
        pathY = list(preprocessed.cleanDLC['Spine_y'][start-length:start])
        pathY = [0-coor for coor in pathY] #Y coordinates are flipped for some reason
        segs = [[(pathX[i],pathY[i]),(pathX[i+1],pathY[i+1])] for i in range(len(pathX)-1)]
        line_segments = LineCollection(segs,linestyles='solid',colors=colorMap[:-1])
        
        
        ax[i].set_xlim(50,550)
        ax[i].set_ylim(-460, 0)
        ax[i].plot(pathX[0],pathY[0],marker='*',markersize=20,color='black')
        ax[i].legend(handles,labels,loc='upper left',fontsize='12')
        ax[i].add_collection(line_segments)
        ax[i].set_xticks=([])
        ax[i].set_yticks=([])
        ax[i].xaxis.set_tick_params(labelbottom=False)
        ax[i].yaxis.set_tick_params(labelleft=False)

        start +=length
    return fig

#%%
def timeTurning(preprocessed):
    turns, category = bins30s(preprocessed,preprocessed.angleJudgements)

    turns = pd.DataFrame(turns).transpose()
    counts = pd.DataFrame()
    for column in turns.columns:
        singleCount = turns[column].value_counts()
        counts = pd.concat([counts,singleCount],axis=1)
        
    counts = counts.fillna(0)
    right = counts.loc['Right_Broad']+counts.loc['Right_Medium']+counts.loc['Right_Sharp']
    left = counts.loc['Left_Broad']+counts.loc['Left_Medium']+counts.loc['Left_Sharp']
    
 

    header = pd.MultiIndex.from_product([['Left','Right'],['Before','During','After']],names=['Direction','Time'])    
        
    Turns = pd.DataFrame([left.iloc[:category[0]],left.iloc[category[0]:category[1]],left.iloc[category[1]:],\
            right.iloc[:category[0]], right.iloc[category[0]:category[1]], right.iloc[category[1]:]]).transpose()
    Turns.columns = header
        
    Turns = Turns.unstack().reset_index()
    Turns.columns = ['Direction','Time','level 2','Frames/30Sec']
    
    xOrder = ['Left','Right']
    hueOrder = ['Before','During','After']
    plot_params = {'data':Turns,'x':'Direction','y':'Frames/30Sec','hue':'Time','order':xOrder,'hue_order':hueOrder}
    
    fig,ax = plt.subplots(1,2)
    fig.set_figwidth(13)
    fig.set_figheight(6)
    annotator = Annotator(ax[0],preprocessed.pairs,**plot_params)
    annotator.configure(test='Mann-Whitney',text_format='star',loc='outside',comparisons_correction='Benjamini-Hochberg')
    sns.boxplot(ax=ax[0],**plot_params,showfliers=False)
    annotator.apply_and_annotate()
    
    right = Turns[Turns['Direction']=='Right']
    left = Turns[Turns['Direction']=='Left']
    xrange = np.array(range(len(right.dropna())))
    ax[1].plot(xrange,right['Frames/30Sec'].dropna(),color='firebrick')
    ax[1].plot(xrange,left['Frames/30Sec'].dropna(),color='forestgreen')
    ax[1].axvline(x=category[0],color='k',linestyle='--')
    ax[1].axvline(x=category[1],color='k',linestyle='--')
    ax[1].set_xlabel('Time (Sec)')
    
    if max(xrange)<13:
        ax[1].set_xticks(xrange)
        ax[1].set_xticklabels(((xrange+1)*30).astype(str))
    else:
        ax[1].set_xticks(xrange)
        lab = ((((xrange+1)%3==0)*xrange+1)*30).astype(str)
        lab[lab=='30']=''
        ax[1].set_xticklabels(lab)
    
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['firebrick','forestgreen']]
    ax[1].legend(handles,['Right','Left'],loc='upper right',fontsize='12')
    
    
    fig.suptitle('Frames Spent Turning')
    
    return fig
#%%
def distTurning(preprocessed):
    distance, category = bins30s(preprocessed,preprocessed.distance)
    turns, category = bins30s(preprocessed,preprocessed.direction)
    
    
    turningList = []
    for segment in range(len(turns)):
        moving = np.where(np.array(distance[segment])>preprocessed.distanceThresh)[0]
        left = np.where(np.array(turns[segment])=='Left')[0]
        right = np.where(np.array(turns[segment])=='Right')[0]
        
        leftMov = [i for i in moving if i in left]
        rightMov = [i for i in moving if i in right]
        
        leftDist = np.array(distance[segment][leftMov]).sum()
        rightDist = np.array(distance[segment][rightMov]).sum()
        
        if segment < category[0]:
            time = 'Before'
        elif segment < category[1]:
            time = 'During'
        else:
            time = 'After'
        turningList.append(['Left',time,leftDist])
        turningList.append(['Right',time,rightDist])
        
    distTurning = pd.DataFrame(turningList,columns=['Direction','Time','Cm Turning/30Sec'])
    
    xOrder = ['Left','Right']
    hueOrder = ['Before','During','After']
    plot_params = {'data':distTurning,'x':'Direction','y':'Cm Turning/30Sec','hue':'Time','order':xOrder,'hue_order':hueOrder}
    
    fig,ax = plt.subplots(1,2)
    fig.set_figwidth(13)
    fig.set_figheight(6)
    annotator = Annotator(ax[0],preprocessed.pairs,**plot_params)
    annotator.configure(test='Mann-Whitney',text_format='star',loc='outside',comparisons_correction='Benjamini-Hochberg')
    sns.boxplot(ax=ax[0],**plot_params,showfliers=False)
    annotator.apply_and_annotate()
    
    
    right = distTurning[distTurning['Direction']=='Right']
    left = distTurning[distTurning['Direction']=='Left']
    xrange = np.array(range(len(right.dropna())))
    ax[1].plot(xrange,right['Cm Turning/30Sec'].dropna(),color='firebrick')
    ax[1].plot(xrange,left['Cm Turning/30Sec'].dropna(),color='forestgreen')
    ax[1].axvline(x=category[0],color='k',linestyle='--')
    ax[1].axvline(x=category[1],color='k',linestyle='--')
    ax[1].set_xlabel('Time (Sec)')
    
    if max(xrange)<13:
        ax[1].set_xticks(xrange)
        ax[1].set_xticklabels(((xrange+1)*30).astype(str))
    else:
        ax[1].set_xticks(xrange)
        lab = ((((xrange+1)%3==0)*xrange+1)*30).astype(str)
        lab[lab=='30']=''
        ax[1].set_xticklabels(lab)
    
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['firebrick','forestgreen']]
    ax[1].legend(handles,['Right','Left'],loc='upper right',fontsize='12')
    
    fig.suptitle('Distance Traveled Turning')
    
    
    return fig
#%%
def overallDist(preprocessed):
    distance, category = bins30s(preprocessed,preprocessed.distance)
    
    distList = []
    for segment in range(len(distance)):
        moving = np.where(np.array(distance[segment])>preprocessed.distanceThresh)[0]
        
        totMove = np.array(distance[segment][moving]).sum()
        if segment < category[0]:
            time = 'Before'
        elif segment < category[1]:
            time = 'During'
        else:
            time = 'After'
        
        distList.append([time,totMove])
    
    distOverall = pd.DataFrame(distList,columns=['Time','Cm/30Sec'])
    
    xOrder = ['Before','During','After']
    plot_params = {'data':distOverall,'x':'Time','y':'Cm/30Sec','order':xOrder}
    
    pairs = [('Before','During'),('During','After'),('Before','After')]
    
    fig,ax = plt.subplots(1,2)
    fig.set_figwidth(13)
    fig.set_figheight(6)
    annotator = Annotator(ax[0],pairs,**plot_params)
    annotator.configure(test='Mann-Whitney',text_format='star',loc='outside')
    sns.boxplot(ax=ax[0],**plot_params,showfliers=False)
    annotator.apply_and_annotate()
    
    
    
    xrange = np.array(range(len(distOverall['Cm/30Sec'])))
    ax[1].plot(xrange,distOverall['Cm/30Sec'].dropna(),color='mediumblue')
    ax[1].axvline(x=category[0],color='k',linestyle='--')
    ax[1].axvline(x=category[1],color='k',linestyle='--')
    ax[1].set_xlabel('Time (Sec)')
    
    if max(xrange)<13:
        ax[1].set_xticks(xrange)
        ax[1].set_xticklabels(((xrange+1)*30).astype(str))
    else:
        ax[1].set_xticks(xrange)
        lab = ((((xrange+1)%3==0)*xrange+1)*30).astype(str)
        lab[lab=='30']=''
        ax[1].set_xticklabels(lab)
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['mediumblue']]
    ax[1].legend(handles,['OverallDist'],loc='upper right',fontsize='12')
    
    fig.suptitle('Distance Traveled in 30 Sec')
    
    return fig
#%%
def duringStimulation(preprocessed):
    stimPattern = 0
    stimOn = []
    stimOff = []
    offset = preprocessed.offset
    j = 0
    for i in preprocessed.stimTimes:
        while stimPattern<i:
            stimOn.append(stimPattern)
            stimPattern+=preprocessed.stimPattern[0]
            
            stimOff.append(stimPattern)
            stimPattern+=preprocessed.stimPattern[1]
        stimPattern = i+offset[j]
        
    stimOn = stimOn[:-1]
    stimOff = stimOff[:-1]
    dWhen = ['Before','During','After']
    stimOn = alignStimulation(preprocessed.cleanDLC.index,stimOn)
    stimOff = alignStimulation(preprocessed.cleanDLC.index,stimOff)
    halfOffTime = int(preprocessed.stimPattern[1]/2)
    
    trial = []
    whenChange = []
    j = 0
    for i in range(len(stimOn)-1):
        if stimOn[i] > preprocessed.shiftTimes[j]:
            whenChange.append((i+1)*3)
            j+=1
        trial.append(['On',np.median(preprocessed.speed[stimOn[i]:stimOff[i]]),dWhen[j]])
        trial.append(['Off1',np.median(preprocessed.speed[stimOff[i]:stimOff[i]+halfOffTime]),dWhen[j]])
        trial.append(['Off2',np.median(preprocessed.speed[stimOff[i]+halfOffTime:stimOff[i]+(halfOffTime*2)]),dWhen[j]])
            
    duringStim = pd.DataFrame(trial)
    duringStim.columns=['Stimulation','Speed Cm/Sec','Time']
    
    xOrder = dWhen
    hueOrder = ['On','Off1','Off2']
    plot_params = {'data':duringStim,'x':'Time','y':'Speed Cm/Sec','hue':'Stimulation','order':xOrder,'hue_order':hueOrder}
    
    fig,ax = plt.subplots(1,2)
    fig.set_figwidth(13)
    fig.set_figheight(6)
    pairs = (
        [('During','On'),('During','Off1')],
        [('During','On'),('During','Off2')],
        [('After','On'),('During','On')],
        [('Before','On'),('During','On')])
    annotator = Annotator(ax[0],pairs,**plot_params)
    annotator.configure(test='Mann-Whitney',text_format='star',loc='outside',comparisons_correction='Benjamini-Hochberg')
    sns.barplot(ax=ax[0],**plot_params,palette=['firebrick','grey','k'])
    annotator.apply_and_annotate()
    
    
    ax[1] = duringStim['Speed Cm/Sec'].plot(kind='bar', \
            color=duringStim['Stimulation'].replace({'On':'firebrick','Off1':'grey','Off2':'k'}),
            xticks=[],)
    ax[1].axvline(x=whenChange[0],color='b',linestyle='--',linewidth=3)
    ax[1].axvline(x=whenChange[1],color='b',linestyle='--',linewidth=3)
    samples = len(duringStim['Speed Cm/Sec'])
    ax[1].set_xticks([samples/6,samples/2,5*samples/6])
    ax[1].set_xticklabels(dWhen)
    ax[1].set_xlabel('Time (Minutes)')
    
    
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['firebrick','black']]
    ax[1].legend(handles,['on','off'],loc='upper right',fontsize='12')
    
    fig.suptitle('Speed During "Stimulation"')
    
def alignStimulation(index,stimTimes):
    shiftTimes = []
    for time in stimTimes:
        diff = np.array(index)-time
        shiftTimes.append(np.where(abs(diff)==min(abs(diff)))[0][0])
    return shiftTimes

#%%
def cumulativeAngle(preprocessed):
    angles = np.array(preprocessed.theta)
    turns = pd.Series(preprocessed.angleJudgements)
    
    
    leftPos = ['Left_Broad','Left_Medium','Left_Sharp']

    angles = 180-angles
    leftWhere = np.where(turns.isin(leftPos))[0]
    leftAngles = 0-angles[leftWhere]
    angles[leftWhere] = leftAngles
    
    filterLow = np.where(turns == 'Straight')[0]
    angles[filterLow] = 0
    
    cumulative = np.cumsum(angles)
    
    slope = [cumulative[i]-cumulative[i-1] for i in range(1,len(cumulative))]
    slope = [0]+slope
    
    slopes,category = bins30s(preprocessed, slope)
    
    z = stats.zscore(slope[:preprocessed.shiftTimes[0]])
    st = np.array(slope[:preprocessed.shiftTimes[0]]).std()
    initBias = z[0]*st
    xrange = np.array(range(len(cumulative)))
    normCumulative = cumulative+(initBias*xrange)

    fig,ax = plt.subplots(3,1)
    plt.subplots_adjust(hspace=1)
    fig.set_figwidth(12)
    fig.set_figheight(9)
    fig.suptitle('Turning Bias')
    ax[0].plot(cumulative)
    ax[0].set_title('Cumulative Turning Angle')
    ax[0].set_ylabel('Cummulative Angle (Right Positive)')
    ax[0].set_xlabel('Frame')
    ax[0].axvline(x=preprocessed.shiftTimes[0],color='k',linestyle='--')
    ax[0].axvline(x=preprocessed.shiftTimes[1],color='k',linestyle='--')
    
    xrange = np.array(range(len(slopes)))
    sns.barplot(ax=ax[1],data=slopes)
    ax[1].set_xlabel('Time (Sec)')
    ax[1].set_ylabel('Average Slope')
    ax[1].set_title('Slope of Turning Bias')
    if max(xrange)<13:
        ax[1].set_xticks(xrange)
        ax[1].set_xticklabels(((xrange+1)*30).astype(str))
    else:
        ax[1].set_xticks(xrange)
        lab = ((((xrange+1)%3==0)*xrange+1)*30).astype(str)
        lab[lab=='30']=''
        ax[1].set_xticklabels(lab)
        
    ax[2].plot(normCumulative)
    ax[2].set_title('Noramlized Cumulative Turning Angle')
    ax[2].set_ylabel('Cummulative Angle (Right Positive)')
    ax[2].set_xlabel('Frame')
    ax[2].axvline(x=preprocessed.shiftTimes[0],color='k',linestyle='--')
    ax[2].axvline(x=preprocessed.shiftTimes[1],color='k',linestyle='--')
    
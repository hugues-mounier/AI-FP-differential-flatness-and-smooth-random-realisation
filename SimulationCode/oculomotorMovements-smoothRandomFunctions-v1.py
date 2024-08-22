# 
# Oculomotor movement tracking simulation -- H Mounier -- July 2024 
#
import matplotlib.pyplot as plt
import matplotlib as matpl
import numpy as np
import scipy.integrate as spy
import sys
from   dataclasses import dataclass
from   abc import ABC, abstractmethod
from   os import chdir, getcwd, path, makedirs
from   datetime import date, datetime

# use latex fonts 
# matpl.style.use('text.usetex') # style not found
matpl.rcParams['text.usetex'] = True
plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Roman"})


# Workaround the bug from which we are not in the current working directory
chdir(path.abspath(path.dirname(__file__)))

########################
## Utilitary functions
########################

def printf(format, *args):
    sys.stdout.write(format % args)

def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)


###############################################
##  parameters
###############################################

class Pars(ABC):
    @abstractmethod
    def get(self):
        pass
    def writeOnFile(self, fileid):
        pass

@dataclass 
class FluctSmoothRandFunPars(Pars):
    L:      float 
    lmbda:  float  
    scale:  float
    trajName:   str
    def get(self):
        return [self.L, self.lmbda, self.scale]
    def printPars(self):
        printf('\nSmooth Random Function parameters: L = %.5g  $\\lambda$ = %.5g  scale = %.5g\n\n', 
                 self.L, self.lmbda, self.scale)
    def writeOnFile(self, filename):
        stream = open(filename, "w+")
        firstLine = "\nSmooth Random function Fluctuation Trajectory of " + self.trajName + " parameters\n"
        fprintf(stream, firstLine); fprintf(stream, "-" * len(firstLine) + "\n")
        fprintf(stream, "Smooth random function length L                   : %.5g\n", self.L)
        fprintf(stream, "Smooth random function wavelength $\\lambda$       : %.5g\n", self.lmbda)
        fprintf(stream, "Smooth random function scale                      : %.5g\n", self.scale )
        stream.close()


@dataclass 
class ZeroTrajPars:
    def get(self):
        return 
    def writeOnFile(self, fileid):
        return

   
##########################################
## Reference and fluctuation trajectories 
##########################################

class Traj(ABC):
    @abstractmethod
    def traj(self, t):
        pass


@dataclass
class SmoothRandFunTraj(Traj):
    L:      float
    lmbda:  float 
    scale:  float 
    r:      float
    a:      np.ndarray[float, np.dtype[np.float64]]
    b:      np.ndarray[float, np.dtype[np.float64]]
    def __init__(self, L, lmbda, scale):
        self.L = L; self.lmbda = lmbda; self.scale = scale
        r = int(np.floor(self.L/self.lmbda)); self.r = r
        var = 1/(2*r+1)
        self.a = np.empty([r+1], dtype = float)
        self.b = np.empty([r+1], dtype = float)
        for i in range(1, r+1):
            self.a[i] = np.random.normal(0, var)
            self.b[i] = np.random.normal(0, var)
    def traj(self, t):
        L = self.L; scale = self.scale
        a = self.a; b = self.b; r = self.r
        pi = np.pi
        zeta = a[0]; dotzeta = 0; ddotzeta = 0
        for j in range(1, r):
            nuj = 2*pi*j/L
            zeta     = zeta     + a[j] * np.cos(nuj*t)        + b[j] * np.sin(nuj*t)
            dotzeta  = dotzeta  - a[j]*nuj * np.sin(nuj*t)    + b[j]*nuj * np.cos(nuj*t)
            ddotzeta = ddotzeta - a[j]*nuj**2 * np.cos(nuj*t) - b[j]*nuj**2 * np.sin(nuj*t)
        return [scale*zeta, scale*dotzeta, scale*ddotzeta]

@dataclass
class ZeroTraj(Traj):
    def traj(self, t):
        return [0*t, 0*t, 0*t]


######################
## Plotting functions
######################

def indiviualPlotAndSave(fid, ttoPlot, xToPlot, 
                  xLabelStr, yLabelStr, titleStr, saveFigStr, plotsDirectory, figNameEnd,
                  firstLast = "first&last", lineThickness = 1.5, lineColor = "blue"):
        plt.figure(fid);     
        if ("first" in firstLast):
            plt.cla(); plt.clf();
        plt.plot(ttoPlot, xToPlot, linewidth = lineThickness, color = lineColor)
        if ("last" in firstLast):
            plt.xlabel(xLabelStr); plt.ylabel(yLabelStr);
            plt.title(titleStr);
            figName = plotsDirectory + '/' + saveFigStr % (figNameEnd)
            plt.grid(); plt.savefig(figName, format="pdf");
            printf('Figure %s saved\n', saveFigStr % (figNameEnd))
        fid = fid + 1;
        return fid


def createFluctuationTrajs(trajType):
    # default parameter values
    L = 8; lmbda = 2; scale = 1e-5;   # fluctPars
    match trajType:
            case s if 'SRFTiSc' in s:  # Smmoth Random Function Tiny Scale
            # r = floor(L/lmbda) - 2*pi*j/L = N*2*pi <=> L = j/N ; j = 1, ..., r
                L = 8; lmbda = 2; scale = 2e-6;   # fluctPars
            case s if 'SRFSmSc' in s:  # Smmoth Random Function Small Scale
                L = 8; lmbda = 2; scale = 1e-1;   # fluctPars
            case s if 'SRFMdSc' in s:  # Smmoth Random Function Medium Scale
                L = 8; lmbda = 2; scale = 1;   # fluctPars
            case s if 'SRFBgSc' in s:    # Smmoth Random Function Big Scale
                L = 8; lmbda = 2; scale = 5;   # fluctPars
            case s if 'SRFVBgSc' in s:  # Smmoth Random Function Very Big Scale
                L = 8; lmbda = 2; scale = 10;   # fluctPars
            case s if 'SRFHgSc' in s:    # Smmoth Random Function Huge Scale
                L = 8; lmbda = 2; scale = 100;   # fluctPars
            case s if 'SRFLowLambda' in s:
                L = 8; lmbda = 0.1; scale = 1;   # fluctPars
            case default:
                print(trajType)
                print('No matching fluctuation type; default values taken')
    printf('Trajectory type : %s -- L = %.5g  lmbda = %.5g  scale = %.5g\n', trajType, L, lmbda, scale)
    fluctTrajPars = FluctSmoothRandFunPars(L, lmbda, scale, "abscissa x"); 
    fluctTraj    = SmoothRandFunTraj(L, lmbda, scale)
    fluctTrajPars.printPars()
    return [fluctTrajPars, fluctTraj]
   

def performTask(whatToDo, refTrajType, refFluctType, saveFigures):
    # directory and file names $$$
    plotsDirectory = './PlotsFromSimulations'
    if not path.exists(plotsDirectory):
        makedirs('./PlotsFromSimulations')
    t = np.arange(0, 50, 0.01); fid = 1


    # Plot no 1
    refFluctType = 'SRFMdSc'
    figNameEnd   = 'smoothRandFun-' +  whatToDo  + '-' + refFluctType 
    paramsFilename = plotsDirectory + '/pars-' + figNameEnd + ".txt"
    firstLast = "first&last"; lineThick = 1.5; lineColor = "blue"

    [fluctTrajPars, fluctTraj] = createFluctuationTrajs(refFluctType)
    [xFluct, dotxFluct, ddotxFluct] = fluctTraj.traj(t); 
    fid = indiviualPlotAndSave(fid, t, xFluct, 'Time (s)', '$\\zeta (t)$', 'Fluctuation trajectory', 
                                "zeta-%s.pdf", plotsDirectory, figNameEnd, firstLast, lineThick, lineColor)
    fluctTrajPars.writeOnFile(paramsFilename)

    # Plot no 2
    refFluctType = 'SRFLowLambda'
    figNameEnd   = 'smoothRandFun-' +  whatToDo + '-' + refFluctType 
    paramsFilename = plotsDirectory + '/pars-' + figNameEnd + ".txt"
    [fluctTrajPars, fluctTraj] = createFluctuationTrajs(refFluctType)
    [xFluct, dotxFluct, ddotxFluct] = fluctTraj.traj(t); 
    fid = indiviualPlotAndSave(fid, t, xFluct, 'Time (s)', '$\\zeta (t)$', 'Fluctuation trajectory', 
                                "zeta-%s.pdf", plotsDirectory, figNameEnd, firstLast, lineThick, lineColor)
    fluctTrajPars.writeOnFile(paramsFilename)
    



def main():
    # reference traj, fluctuation and action law choice
    saveFigures  = True                 # Save or plot
    taskId = 1
    ctrlType = 'flatCtrlLaw'; refTrajType  = 'tanhTr'; 
    # Open loop control computation and plot, no fluctuation
    whatToDo = 'Task-' + str(taskId) + '-plotFluct';  refFluctType = 'SRFSmSc';    
    performTask(whatToDo, refTrajType, refFluctType, saveFigures)
    taskId = taskId + 1
    return

main()



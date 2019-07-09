from mpi4py import MPI
import numpy as np
from netCDF4 import MFDataset, num2date
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
import os
import datetime
from sys import argv
import warnings
warnings.filterwarnings('ignore')

def loadData(BEGINDATE):
    BEGINDATE = datetime.datetime(1915,int(BEGINDATE[:2]),int(BEGINDATE[2:]))
    #increment by 13 and then do <= when finding dates that match to get 14-day sum
    ENDDATE = BEGINDATE + datetime.timedelta(days=13)
    if (ENDDATE.month == 2) and (ENDDATE.day == 29):
        ENDDATE += datetime.timedelta(days=1)
    YEAR = 1915

    path = '/home/tdickinson/data/Livneh/'
    files = os.listdir(path)
    files = [path+i for i in files]

    nc = MFDataset(files,'r')
    times = nc.variables['time'][:]
    timeUnits = nc.variables['time'].units
    try: timeCalendar = nc.variables['time'].calendar
    except: timeCalendar = 'standard'

    times = num2date(times,timeUnits,timeCalendar)
    years = np.array([d.year for d in times])
    months = np.array([d.month for d in times])
    days = np.array([d.day for d in times])

    if BEGINDATE.month == ENDDATE.month:
        locs = np.where(((months==BEGINDATE.month)&(days>=BEGINDATE.day)) & ((months==ENDDATE.month)&(days<=ENDDATE.day)))[0]
    else:
        locs = np.where(((months==BEGINDATE.month)&(days>=BEGINDATE.day)) | ((months==ENDDATE.month)&(days<=ENDDATE.day)))[0]

    #I have 97 years in total, so locs should have size of 1358
    if locs.size != 1358:
        leapDay = np.where((months==2)&(days==29))[0]
        leapDayLocs = [np.where(locs==i)[0][0] for i in leapDay]
        locs = np.delete(locs,leapDayLocs)

    precip = nc.variables['prec'][locs,:,:]
    precip = precip.filled(np.nan)
    nc.close()
    #get 14-day sum for each year
    t,y,x = precip.shape
    precip = np.sum(precip.reshape(97,t/97,y,x),axis=1)
    t,y,x = precip.shape
    precip = precip.reshape(t,y*x)
    nonNaN = np.where(~np.isnan(precip[0,:]))[0]
    return precip[:,nonNaN], precip.shape[1], nonNaN

def main():
    #initialize communicator, get rank of each thread, and get total number of threads
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    years = sm.add_constant(np.arange(1915,2012,1))
    BEGINDATE_STR = argv[1]
    qUser = float(argv[2])
    precip = None
    totalGrids = None
    nonNaN = None
    allSlopes = None
    allIntercepts = None

    if rank == 0:
        precip, totalGrids, nonNaN = loadData(BEGINDATE_STR)
        precip = np.array_split(precip,size,axis=1)

    data = comm.scatter(precip,root=0)
    numOfSlopes = data.shape[1]

    slopes = np.zeros((numOfSlopes,))*np.nan
    intercepts = np.zeros((numOfSlopes,))*np.nan
    for i in range(numOfSlopes):
        mod = QuantReg(exog=years,endog=data[:,i]).fit(q=qUser)
        intercepts[i] = mod.params[0]
        slopes[i] = mod.params[1]

    #get number of slopes from each thread
    counts = comm.gather(sendobj=numOfSlopes,root=0)
    #broadcast so that each thread has the list
    counts = comm.bcast(counts,root=0)

    #now, calculate the displacements
    dspls = [0]
    #dspls will be a list specifying where each thread's data begins
    for i in range(len(counts)-1):
        #a given thread's data will start where one earlier thread's data ended
        dspls.append(dspls[i]+counts[i])

    #setup holder array in the 0 thread for aggregated data
    if rank == 0:
        allSlopes = np.empty(sum(counts),dtype=np.float64)
        allIntercepts = np.empty(sum(counts),dtype=np.float64)

    #send the data in a given thread which has counts[rank] data
    sendbufSlopes = [slopes,counts[rank]]
    sendbufIntercepts = [intercepts,counts[rank]]
    #specify where the data should be gathered into
    #ORDER: list/array to store in, counts, displacements, dtype
    recvbufSlopes = [allSlopes,counts,dspls,MPI.DOUBLE]
    recvbufIntercepts = [allIntercepts,counts,dspls,MPI.DOUBLE]
    comm.Gatherv(sendbufSlopes,recvbufSlopes,root=0)
    comm.Gatherv(sendbufIntercepts,recvbufIntercepts,root=0)
    if rank == 0:
        outputSlopes = np.zeros((totalGrids,))*np.nan
        outputIntercepts = np.zeros((totalGrids,))*np.nan
        outputSlopes[nonNaN] = allSlopes
        outputIntercepts[nonNaN] = allIntercepts
        np.save('/home/tdickinson/data/windows/slopes/models_14_%s_begin_%s'%(qUser,BEGINDATE_STR),outputSlopes)
        np.save('/home/tdickinson/data/windows/intercepts/models_14_%s_begin_%s'%(qUser,BEGINDATE_STR),outputIntercepts)
        print('%s done'%BEGINDATE_STR)

if __name__ == '__main__':
    main()

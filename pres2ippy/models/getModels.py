from mpi4py import MPI
import numpy as np
from netCDF4 import MFDataset, num2date
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
import os
import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

class InputError(Exception):
    pass

def loadData(BEGINDATE, length):
    BEGINDATE = datetime.datetime(1915,int(BEGINDATE[:2]),int(BEGINDATE[2:]))
    #example: increment by 13 and then do <= when finding dates that match to get 14-day sum
    ENDDATE = BEGINDATE + datetime.timedelta(days=length-1)
    if (ENDDATE.month == 2) and (ENDDATE.day == 29):
        ENDDATE += datetime.timedelta(days=1)

    #Livneh runs from 1915 through 2011 so 97 years unless considering December 19 through December 31
    #since it runs into next year, so I must ignore the Jan. 1915 days and Dec. 2011 days
    #FIXME: adjust day to be compatible for any input length (currently hard coded for 14 day length)
    totalYears = 96 if (BEGINDATE.month == 12) and (BEGINDATE.day >= 19) else 97

    path = '/home/tdickinson/data/Livneh/'
    files = os.listdir(path)
    files.sort()
    files = [path+i for i in files]

    nc = MFDataset(files,'r')
    times = nc.variables['time'][:]
    timeUnits = nc.variables['time'].units
    try: timeCalendar = nc.variables['time'].calendar
    except: timeCalendar = 'standard'

    times = num2date(times,timeUnits,timeCalendar)
    months = np.array([d.month for d in times])
    days = np.array([d.day for d in times])

    if BEGINDATE.month == ENDDATE.month:
        locs = np.where(((months==BEGINDATE.month)&(days>=BEGINDATE.day)) & ((months==ENDDATE.month)&(days<=ENDDATE.day)))[0]
    else:
        locs = np.where(((months==BEGINDATE.month)&(days>=BEGINDATE.day)) | ((months==ENDDATE.month)&(days<=ENDDATE.day)))[0]

    #check to ensure the correct number of days (i.e., remove leap days or beginning January days in late December start)
    if locs.size != 14*totalYears:
        if BEGINDATE.month == 2:
            leapDay = np.where((months==2)&(days==29))[0]
            badLocs = [np.where(locs==i)[0][0] for i in leapDay]
        else:
            #discard the Jan. 1915 locs and Dec. 2011 locs
            daysInDecember = 31 - BEGINDATE.day + 1
            daysInJanuary = ENDDATE.day
            badLocs = np.append(locs[:daysInJanuary], locs[-daysInDecember:])
            badLocs = [np.where(locs==i)[0][0] for i in badLocs]
        locs = np.delete(locs,badLocs)

    precip = nc.variables['prec'][locs,:,:]
    precip = precip.filled(np.nan)
    nc.close()
    #get 14-day sum for each year
    t,y,x = precip.shape
    precip = np.sum(precip.reshape(totalYears, length, y, x), axis=1)
    t,y,x = precip.shape
    precip = precip.reshape(t,y*x)
    nonNaN = np.where(~np.isnan(precip[0,:]))[0]
    years = sm.add_constant(np.arange(1915, 1915+totalYears, 1))

    return precip[:,nonNaN], precip.shape[1], nonNaN, years

def main():
    #unpack command line arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quantile", type=float, help="quantile for regression model")
    parser.add_argument("-b", "--begin", type=str, help="begin date for window")
    parser.add_argument("-l", "--length", type=int, help="length of window")
    args = parser.parse_args()

    qUser = args.quantile
    BEGINDATE_STR = args.begin
    LENGTH = args.length

    if qUser is None:
        raise InputError('No quantile to fit model was input.')

    if LENGTH is None:
        raise InputError('No window length was input.')

    if BEGINDATE_STR is None:
        raise InputError('No beginning date was input.')

    #initialize communicator, get rank of each thread, and get total number of threads
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    precip = None
    totalGrids = None
    nonNaN = None
    years = None
    allSlopes = None
    allIntercepts = None

    if rank == 0:
        precip, totalGrids, nonNaN, years = loadData(BEGINDATE_STR, LENGTH)
        precip = np.array_split(precip,size,axis=1)

    data = comm.scatter(precip, root=0)
    years = comm.bcast(years, root=0)
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

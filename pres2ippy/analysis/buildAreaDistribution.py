from mpi4py import MPI
import rainpy as rp
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--month", type=str, help="month to analyze")
    parser.add_argument("-a", "--area", type=int, help="areal threshold")
    parser.add_argument("-l", "--length", type=int, help="event length")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to use")
    args = parser.parse_args()

    MONTH = args.month
    AREA = args.area
    LENGTH = args.length
    DATASET = args.dataset

    validMonths = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    validDatasets = ['livneh', 'era5', 'era5_1', 'era5_2', 'era5_3']

    if MONTH is None:
        raise ValueError('Month not input.')
    elif MONTH.lower() not in validMonths:
        raise ValueError('Invalid month input. Valid inputs are %s'%(validMonths))

    if DATASET is None:
        raise ValueError('Dataset not input.')
    elif DATASET.lower() not in validDatasets:
        raise ValueError('Invalid dataset input. Valid datasets are %s.'%(validDatasets))

    #results from buildKDEDistribution.py
    if DATASET == 'livneh':
        DEFAULT_TICK = 0.0865
    elif DATASET == 'era5':
        DEFAULT_TICK = 0.1808
    elif DATASET == 'era5_1':
        DEFAULT_TICK = 0.1517
    elif DATASET == 'era5_2':
        DEFAULT_TICK = 0.2352
    else:
        DEFAULT_TICK = 0.2547

    usrMonth = validMonths.index(MONTH.lower())

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    days = days[usrMonth]
    years = None
    allAreas = None

    if rank == 0:
        if DATASET == 'livneh':
            years = np.arange(1915,2019,1)
        elif 'era5' in DATASET:
            years = np.arange(1979,2019,1)
        years = np.array_split(years, size)

    years = comm.scatter(years, root=0)

    areas = []
    for i in years:
        for j in range(1,days+1):
            #run through the analysis but stop after calculating area
            date = rp.DateTest(month=usrMonth+1, day=j, year=i, length=LENGTH, dataset=DATASET)
            date.kde()
            perc = date.calcKDEPercentile(90)
            if perc >= DEFAULT_TICK:
                date.getContours(ticks=perc)
            else:
                date.getContours(ticks=DEFAULT_TICK)
            date.calcAreas(areaThreshold=AREA)
            if len(date.polys) > 0:
                areas.append([num for num in date.areas[list(date.areas.keys())[0]] if num >= AREA])
            del date

    #flatten list
    areas = np.array([item for sublist in areas for item in sublist])

    #get number of total data points from each cpu and broadcast to all
    countArea = comm.gather(sendobj=areas.shape[0],root=0)
    countArea = comm.bcast(countArea,root=0)

    #calc displacements
    dspls = [0]
    for i in range(len(countArea)-1):
        dspls.append(dspls[i]+countArea[i])

    #setup holder array for aggregated data
    if rank == 0:
        allAreas = np.empty(sum(countArea),dtype=np.float64)

    #receive data
    sendbufArea = [areas,countArea[rank]]
    recvbufArea = [allAreas,countArea,dspls,MPI.DOUBLE]
    comm.Gatherv(sendbufArea, recvbufArea, root=0)
    if rank == 0:
        np.save(f'/scratch/tdickinson/area/{DATASET}.{LENGTH}.areas.{validMonths[usrMonth])}.npy', allAreas)

if __name__ == '__main__':
    main()

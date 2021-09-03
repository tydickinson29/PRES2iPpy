#################################################################################
#Uses rainpy.py to build a distribution of KDE densities for all length-day windows
#for an input month across all years in dataset period of record. Saves distribution
#as one .npy file. MPI is implemented so the script can be run in parallel. We use
#the 12 files (one for each month) to build the entire KDE distribution to find the
#99th percentile to use as the bounding contour for extreme polygons.

#Arguments
#---------
#month : str
#   3 letter abbreviation for month to analyze.
#length : int
#   Total number of days in the window.
#dataset : str
#   Dataset to use for analysis.

#Author : Ty A. Dickinson
#Last Updated : June 2021
#################################################################################

from mpi4py import MPI
import rainpy as rp
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--month", type=str, help="month to analyze")
    parser.add_argument("-l", "--length", type=int, help="event length")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to use")
    args = parser.parse_args()

    MONTH = args.month
    LENGTH = args.length
    DATASET = args.dataset

    validMonths = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    validDatasets = ['livneh', 'era5', 'era5_1', 'era5_2', 'era5_3']

    if MONTH is None:
        raise ValueError('Month not input.')
    elif MONTH.lower() not in validMonths:
        raise ValueError(f'Invalid month input. Valid inputs are {validMonths}')

    if DATASET is None:
        raise ValueError('Dataset not input.')
    elif DATASET.lower() not in validDatasets:
        raise ValueError(f'Invalid dataset input. Valid datasets are {validDatasets}.')

    usrMonth = validMonths.index(MONTH.lower())

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    days = days[usrMonth]
    years = None
    allDensities = None

    if rank == 0:
        if DATASET == 'livneh':
            years = np.arange(1915,2019,1)
        else:
            pass
        years = np.array_split(years, size)

    years = comm.scatter(years, root=0)
    gridSize = rp.DateTest(month=1, day=1, year=years[0], length=14, dataset=DATASET).kdeGridX.size

    densities = np.zeros((days*len(years), gridSize))*np.nan
    counter = 0
    for yr in years:
        for d in range(1,days+1):
            #run through the analysis but stop after computing KDE
            date = rp.DateTest(month=usrMonth+1, day=d, year=yr, length=LENGTH, dataset=DATASET)
            date.kde()
            densities[counter,:] = date.density.flatten()
            counter += 1
            del date

    #just finding a percentile so shape doesn't matter
    densities = densities.flatten()

    #get number of total data points from each cpu and broadcast to all
    counts = comm.gather(sendobj=densities.shape[0],root=0)
    counts = comm.bcast(counts,root=0)

    #calc displacements
    dspls = [0]
    for i in range(len(counts)-1):
        dspls.append(dspls[i]+counts[i])

    #setup holder array for aggregated data
    if rank == 0:
        allDensities = np.empty(sum(counts),dtype=np.float64)

    #receive data
    sendbuf = [densities,counts[rank]]
    recvbuf = [allDensities,counts,dspls,MPI.DOUBLE]
    comm.Gatherv(sendbuf,recvbuf,root=0)

    if rank == 0:
        np.save(f'/scratch/tdickinson/kde/{DATASET}.{LENGTH}.kde.{validMonths[usrMonth]}.npy', allDensities)

if __name__ == '__main__':
    main()

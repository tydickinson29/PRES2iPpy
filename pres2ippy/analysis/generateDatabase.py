from mpi4py import MPI
import rainpy as rp
import pandas as pd
import numpy as np
import argparse

class InputError(Exception):
    pass

def rounder(data):
    return [np.around(i,decimals=2) for i in data]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--month", type=str, help="month to analyze")
    parser.add_argument("-a", "--area", type=int, help="areal extent to nearest square kilometer")
    parser.add_argument("-l", "--length", type=int, help="event length")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to use")
    args = parser.parse_args()

    MONTH = args.month
    AREA = args.area
    LENGTH = args.length
    DATASET = args.dataset
    DEFAULT_TICK = 0.2710

    validMonths = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    validDatasets = ['livneh', 'era5', 'era5_1', 'era5_2', 'era5_3']

    if MONTH is None:
        raise InputError('Month not input.')
    elif MONTH.lower() not in validMonths:
        raise InputError('Invalid month input. Valid inputs are %s.'%(validMonths))

    if DATASET is None:
        raise InputError('Dataset not input.')
    elif DATASET.lower() not in validDatasets:
        raise InputError('Invalid dataset input. Valid datasets are %s.'%(validDatasets))

    usrMonth = validMonths.index(MONTH.lower())

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    days = days[usrMonth]
    years = None
    allPolys = None

    if rank == 0:
        years = np.arange(1915,2019,1)
        years = np.array_split(years, size)

    years = comm.scatter(years, root=0)

    begin = []
    end = []
    polys = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    centroidX = []
    centroidY = []
    area = []
    areaAvgRain = []
    totalOverThresh = []
    maxTotalRain = []
    maxDailyRain = []
    for i in years:
        for j in range(1,days+1):
            date = rp.DateTest(month=usrMonth+1, day=j, year=i, length=LENGTH, dataset=DATASET)
            date.getObs()
            date.getExtremePoints()
            date.kde()
            date.getContours(ticks=DEFAULT_TICK)
            date.calcAreas(areaThreshold=AREA)
            date.maskOutsideRegion()

            if len(date.polys) > 0:
                begin.append(['%s/%s/%s'%(date.month, date.day, date.year)] * len(date.polys))
                end.append(['%s/%s/%s'%(date.DATE_END.month, date.DATE_END.day, date.DATE_END.year)] * len(date.polys))
                polys.append(date.polys)
                minx.append([d.bounds[0] for d in date.polys])
                miny.append([d.bounds[1] for d in date.polys])
                maxx.append([d.bounds[2] for d in date.polys])
                maxy.append([d.bounds[3] for d in date.polys])
                centroidX.append([i.centroid.x for i in date.polys])
                centroidY.append([i.centroid.y for i in date.polys])
                area.append([num for num in date.areas[list(date.areas.keys())[0]] if num >= AREA])
                areaAvgRain.append(date.weightedTotal)
                totalOverThresh.append(np.nansum(date.regionsDiff, axis=-1))
                maxTotalRain.append(np.nanmax(date.regionsTotal, axis=-1))
                maxDailyRain.append(np.nanmax(date.regionsDaily, axis=-1))
            del date

    begin = [item for sublist in begin for item in sublist]
    end = [item for sublist in end for item in sublist]
    polys = [item for sublist in polys for item in sublist]
    minx = [item for sublist in minx for item in sublist]
    miny = [item for sublist in miny for item in sublist]
    maxx = [item for sublist in maxx for item in sublist]
    maxy = [item for sublist in maxy for item in sublist]
    centroidX = [item for sublist in centroidX for item in sublist]
    centroidY = [item for sublist in centroidY for item in sublist]
    area = [item for sublist in area for item in sublist]
    areaAvgRain = [item for sublist in areaAvgRain for item in sublist]
    totalOverThresh = [item for sublist in totalOverThresh for item in sublist]
    maxTotalRain = [item for sublist in maxTotalRain for item in sublist]
    maxDailyRain = [item for sublist in maxDailyRain for item in sublist]

    cols = ['Begin_Date', 'End_Date', 'Area', 'Area_Averaged_Precip', 'Total_Over_Extreme', 'Maximum_Total_Precip',
            'Maximum_1_Day_Precip', 'Min_Lon', 'Min_Lat', 'Max_Lon', 'Max_Lat', 'Centroid_Lon', 'Centroid_Lat', 'geometry']
    df = pd.DataFrame({'Begin_Date':begin, 'End_Date':end, 'Area':rounder(area),
                       'Area_Averaged_Precip':rounder(areaAvgRain), 'Total_Over_Extreme':rounder(totalOverThresh),
                       'Maximum_Total_Precip':rounder(maxTotalRain),'Maximum_1_Day_Precip':rounder(maxDailyRain),
                       'Min_Lon':rounder(minx), 'Min_Lat':rounder(miny),
                       'Max_Lon':rounder(maxx), 'Max_Lat':rounder(maxy),
                       'Centroid_Lon':rounder(centroidX), 'Centroid_Lat':rounder(centroidY), 'geometry':polys},
                       columns=cols)

    newDF = comm.gather(df, root=0)
    if rank == 0:
        newDF = pd.concat(newDF, ignore_index=True)
        newDF.to_csv(f'/scratch/tdickinson/database/{DATASET}.database.{LENGTH}.{validMonths[usrMonth]}.csv', index=False)

if __name__ == '__main__':
    main()

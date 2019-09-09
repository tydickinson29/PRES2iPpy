from mpi4py import MPI
import rainpy as rp
import pandas as pd
import numpy as np
import geopandas

def rounder(data):
    return [np.around(i,decimals=2) for i in data]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    years = None
    allPolys = None

    if rank == 0:
        years = np.arange(1915, 1915+size, 1)
        print("Last year analyzed: %d"%(years[-1]))
        years = np.array_split(years, size)

    years = comm.bcast(years, root=0)

    begin = []
    end = []
    polys = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    area = []
    areaAvgRain = []
    totalOverThresh = []
    maxTotalRain = []
    maxDailyRain = []
    for i in years:
        for j in months:
            for k in range(1,days[j-1]+1):
                date = rp.DateTest(month=j, day=k, year=i)
                date.kde()
                date.getContours(ticks=0.14)
                date.calcAreas()
                date.maskOutsideRegion()

                if len(date.polys) > 0:
                    begin.append(['%s/%s/%s'%(date.month, date.day, date.year)] * len(date.polys))
                    end.append(['%s/%s/%s'%(date.DATE_END.month, date.DATE_END.day, date.DATE_END.year)] * len(date.polys))
                    polys.append(date.polys)
                    minx.append([i.bounds[0] for i in date.polys])
                    miny.append([i.bounds[1] for i in date.polys])
                    maxx.append([i.bounds[2] for i in date.polys])
                    maxy.append([i.bounds[3] for i in date.polys])
                    area.append([num for num in date.areas[list(date.areas.keys())[0]] if num >= 100000])
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
    area = [item for sublist in area for item in sublist]
    areaAvgRain = [item for sublist in areaAvgRain for item in sublist]
    totalOverThresh = [item for sublist in totalOverThresh for item in sublist]
    maxTotalRain = [item for sublist in maxTotalRain for item in sublist]
    maxDailyRain = [item for sublist in maxDailyRain for item in sublist]

    cols = ['Begin_Date', 'End_Date', 'Area', 'Area_Averaged_Precip', 'Total_Over_Extreme', 'Maximum_Total_Precip',
            'Maximum_1_Day_Precip', 'Min_Lon', 'Min_Lat', 'Max_Lon', 'Max_Lat']
    df = pd.DataFrame({'Begin_Date':begin, 'End_Date':end, 'Area':rounder(area),
                       'Area_Averaged_Precip':rounder(areaAvgRain), 'Total_Over_Extreme':rounder(totalOverThresh),
                       'Maximum_Total_Precip':rounder(maxTotalRain),'Maximum_1_Day_Precip':rounder(maxDailyRain),
                       'Min_Lon':rounder(minx), 'Min_Lat':rounder(miny),
                       'Max_Lon':rounder(maxx), 'Max_Lat':rounder(maxy)},
                       columns=cols)
    gdf = geopandas.GeoDataFrame(df, geometry=polys)

    newDF = comm.gather(df, root=0)
    newGDF = comm.gather(gdf, root=0)
    if rank == 0:
        newDF = pd.concat(newDF, ignore_index=True)
        newDF.to_csv('databasePart1.csv', index=False)
        newGDF = geopandas.GeoDataFrame(pd.concat(newGDF), ignore_index=True)
        newGDF.to_file('databasePart1.shp')


if __name__ == '__main__':
    main()

import rainpy as rp
import pandas as pd
import numpy as np
#import time
#from tqdm import tqdm
#import geopandas

#Testing on sample of dates
months=[12,12,12,12,12,12,12,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
days=[25,26,27,28,29,30,31,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
years=[1998,1998,1998,1998,1998,1998,1998,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999]

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
for i in range(len(months)):
    print(days[i])
    date = rp.DateTest(month=months[i], day=days[i], year=years[i])
    date.kde()
    date.getContours(ticks=0.2)
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

def rounder(data):
    return [np.around(i,decimals=2) for i in data]

cols = ['Begin_Date', 'End_Date', 'Area', 'Area_Averaged_Precip', 'Total_Over_Extreme', 'Maximum_Total_Precip',
        'Maximum_1_Day_Precip', 'Min_Lon', 'Min_Lat', 'Max_Lon', 'Max_Lat']
df = pd.DataFrame({'Begin_Date':begin, 'End_Date':end, 'Area':rounder(area),
                   'Area_Averaged_Precip':rounder(areaAvgRain), 'Total_Over_Extreme':rounder(totalOverThresh),
                   'Maximum_Total_Precip':rounder(maxTotalRain),'Maximum_1_Day_Precip':rounder(maxDailyRain),
                   'Min_Lon':rounder(minx), 'Min_Lat':rounder(miny),
                   'Max_Lon':rounder(maxx), 'Max_Lat':rounder(maxy)},
                   columns=cols)
#df = df.astype({'Begin_Date':'datetime64[ns]', 'End_Date':'datetime64[ns]'})
df.to_csv('database.csv', index=False)

"""
df.to_html('database.html')

gdf = geopandas.GeoDataFrame(df, geometry=polys)
gdf.to_file('database.shp')

months = [1,2,3,4,5,6,7,8,9,10,11,12]
daysInMonth = [31,28,31,30,31,30,31,31,30,31,30,31]

for i in months:
    for j in range(daysInMonth[i-1]):
        obj = rp.DateTest(month=i day=j+1, year=1915)
"""

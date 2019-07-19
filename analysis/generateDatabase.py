import rainpy as rp
import pandas as pd
import time

begin = []
end = []
polys = []
minx = []
miny = []
maxx = []
maxy = []
area = []
for i in range(1,32,1):
    #t1 = time.time()
    date = rp.DateTest(month=1, day=i, year=1999)
    date.getObs()
    date.checkRainyDays()
    date.check3DayTotals()
    date.kde()
    date.getContours()
    date.calcAreas()
    #print('Time elapsed for day %s:%s'%(i, time.time()-t1))

    if len(date.polys) > 0:
        begin.append(['%s/%s/%s'%(date.month, date.day, date.year)]*len(date.polys))
        end.append(['%s/%s/%s'%(date.DATE_END.month,date.DATE_END.day,date.DATE_END.year)]*len(date.polys))
        polys.append(date.polys)
        minx.append([i.bounds[0] for i in date.polys])
        miny.append([i.bounds[1] for i in date.polys])
        maxx.append([i.bounds[2] for i in date.polys])
        maxy.append([i.bounds[3] for i in date.polys])
        area.append([num for num in date.areas[date.areas.keys()[0]] if num >= 100000])

begin = [item for sublist in begin for item in sublist]
end = [item for sublist in end for item in sublist]
polys = [item for sublist in polys for item in sublist]
minx = [item for sublist in minx for item in sublist]
miny = [item for sublist in miny for item in sublist]
maxx = [item for sublist in maxx for item in sublist]
maxy = [item for sublist in maxy for item in sublist]
area = [item for sublist in area for item in sublist]

cols = ['Begin_Date','End_Date','Area','Min_Lon','Min_Lat','Max_Lon','Max_Lat']
df = pd.DataFrame({'Begin_Date':begin, 'End_Date':end, 'Area':area,
                   'Min_Lon':minx, 'Min_Lat':miny, 'Max_Lon':maxx, 'Max_Lat':maxy},
                   columns=cols)
df = df.astype({'Begin_Date':'datetime64[ns]', 'End_Date':'datetime64[ns]'})
df.to_csv('database.csv')

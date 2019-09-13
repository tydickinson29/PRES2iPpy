import numpy as np
import pandas as pd
import geopandas
import shapely.wkt

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on Earth"""

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 #radius of Earth in km
    return c*r

def groupEvents(df, globalCounter):
    global SEARCHING
    localCounter = 1
    try:
        targetRow = df.iloc[globalCounter]
    except IndexError: #signals end of dataframe
        SEARCHING = False
        return

    goodLocs = [globalCounter]
    currentDate = targetRow.Begin_Date
    while True:
        try:
            imon = df.iloc[globalCounter + localCounter]
        except IndexError:
            break

        if (imon.Begin_Date - currentDate).days == 0:
            localCounter += 1
            continue
        elif (imon.Begin_Date - currentDate).days > 1:
            break
        else:
            dist = haversine(lon1=targetRow.Centroid_Lon, lat1=targetRow.Centroid_Lat,
                             lon2=imon.Centroid_Lon, lat2=imon.Centroid_Lat)
            if dist <= 650:
                #keep event if within 1 day and within 650 km
                goodLocs.append(globalCounter + localCounter)
                localCounter += 1
                currentDate = imon.Begin_Date
            else:
                if imon.Begin_Date == df.iloc[globalCounter + localCounter + 1].Begin_Date:
                    localCounter += 1
                    continue
                else:
                    break
    return goodLocs

df = pd.read_csv('database.1915.csv')
df = df.astype({'Begin_Date':'datetime64[ns]', 'End_Date':'datetime64[ns]'})
SEARCHING = True

for imon in range(len(df)):
    goodLocs = groupEvents(df, globalCounter=imon)
    if SEARCHING:
        subset = df.iloc[goodLocs]
        #drop all events except that with largest total over extreme
        oneToChoose = subset['Total_Over_Extreme'].idxmax()
        goodLocs.remove(oneToChoose)
        df.drop(goodLocs, inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        break

#convert to shapefile
df = df.astype({'Begin_Date':str,'End_Date':str})
polys = [shapely.wkt.loads(i) for i in df.geometry]
gdf = geopandas.GeoDataFrame(df, geometry=polys)
gdf.to_file('test.shp')

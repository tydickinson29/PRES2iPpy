import numpy as np
import salem
import xarray as xr
import pandas as pd
import shapely.wkt
from scipy import stats
import glob
import geopandas

def groupEvents(arr, thresh):
    coefs = np.zeros((arr.shape[0]))
    for i in range(coefs.shape[0]):
        coefs[i], _ = stats.pearsonr(x=arr[0,:], y=arr[i,:])
    iloc = np.where(coefs >= thresh)[0]
    return coefs, iloc

def createMaskedArray(df):
    targetRow = df.iloc[0]
    subset = df.iloc[np.where((targetRow.Begin_Date <= df.Begin_Date) & (targetRow.End_Date >= df.Begin_Date))]
    polys = [shapely.wkt.loads(i) for i in subset.geometry]

    lat = np.arange(24, 50.1, 0.1)
    lon = np.arange(232, 294.1, 0.1)
    numRegions = len(subset)

    gridLons, gridLats = np.meshgrid(lon, lat)
    regions = np.ones_like(gridLats)
    regions = np.tile(regions, (numRegions,1)).reshape(numRegions,lat.size,lon.size)

    regions = xr.DataArray(regions, dims=('regions','lat','lon'), coords={'regions':range(numRegions),'lat':lat,'lon':lon})

    regionsMasked = np.ones_like(regions)
    for i in range(numRegions):
        regionsMasked[i,:,:] = regions[i,:,:].salem.roi(geometry=polys[i])

    regionsMasked = np.ma.masked_array(regionsMasked, np.isnan(regionsMasked))
    regionsMasked = regionsMasked.filled(0)

    n,y,x = regionsMasked.shape
    regionsMasked = regionsMasked.reshape(n,y*x)
    return regionsMasked

files = glob.glob('/share/data1/Students/ty/database/*.csv')
df = pd.concat([pd.read_csv(i) for i in files])
df = df.astype({'Begin_Date':'datetime64[ns]', 'End_Date':'datetime64[ns]'})
df = df.sort_values('Begin_Date')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

while True:
    numEvents = len(df)
    events = []
    while len(df) != 0:
        #first create matrix for correlation
        maskedEvents = createMaskedArray(df)

        #calculate correlations and pull out groupedEvents
        coefs, iloc = groupEvents(arr=maskedEvents, thresh=0.5)
        similarEvents = df.iloc[iloc]
        similarEvents.reset_index(drop=True, inplace=True)

        #choose event with largest TOE
        event = similarEvents.iloc[similarEvents['Total_Over_Extreme'].idxmax]
        events.append(event)

        #remove events with correlation >= 0.5
        df.drop(iloc, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(len(df))

    print('\n')
    df = pd.DataFrame(events)
    df = df.sort_values('Begin_Date')
    df.reset_index(drop=True, inplace=True)
    if len(df) == numEvents:
        break

polys = [shapely.wkt.loads(i) for i in df.geometry]
df.to_csv('/share/data1/Students/ty/database/database_14_v1.2.csv', index=False)

df = df.astype({'Begin_Date':str, 'End_Date':str})
gdf = geopandas.GeoDataFrame(df, geometry=polys)
gdf.to_file('/share/data1/Students/ty/database/database_14_v1.2.shp')

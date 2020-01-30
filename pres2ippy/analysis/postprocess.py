import numpy as np
import pandas as pd
import shapely.wkt
import geopandas
import glob

files = glob.glob('/share/data1/ty/database/*.csv')
df = pd.concat([pd.read_csv(i) for i in files])
df = df[df.Area >= 200000]
df = df.astype({'Begin_Date':'datetime64[ns]', 'End_Date':'datetime64[ns]'})
df = df.sort_values('Begin_Date')
df.reset_index(drop=True, inplace=True)
events = []

def findEvent(df):
    """Group events based on first row in preprocessed dataframe. All events that start on or
    before the ending date of the event in the first row are initially considered. Then, events
    are grouped if the interesting area is 50% of either intersectand area. Returns the chosen
    event and the indices of grouped events.
    """
    polys = [shapely.wkt.loads(i) for i in df.geometry]

    #create symmetrical matrix of all intersections in first subset
    intersection = [[i.intersection(j) for j in polys] for i in polys]

    #find fractions relative to both events making intersection; no intersection gives 0
    #rows are relative to one event, columns relative to the other
    fracAreas = np.array([[j.area / polys[i].area for j in intersection[i]] for i in range(len(intersection))])

    #find all polygons that group with the first event then find all polygons that group with all those events
    tmpSim = np.where((fracAreas[0,:] >= 0.5) | (fracAreas[:,0] >= 0.5))[0]
    indices = []
    for i in tmpSim:
        indices.append(np.where((fracAreas[i,:] >= 0.5) | (fracAreas[:,i] >= 0.5))[0])

    indices = np.unique(np.concatenate(indices))
    similarEvents = df.iloc[indices]
    similarEvents.reset_index(drop=True, inplace=True)
    event = similarEvents.iloc[similarEvents['Total_Over_Extreme'].idxmax]
    return event, indices

while len(df) != 0:
    tmpEvents = []
    tmpIndices = []
    tmpDFs = []
    #find first event
    localDF = df.iloc[np.where((df.iloc[0].Begin_Date <= df.Begin_Date) & (df.iloc[0].End_Date >= df.Begin_Date))]
    output = findEvent(localDF)
    tmpEvents.append(output[0])
    tmpIndices.append(output[1])
    tmpDFs.append(localDF.iloc[tmpIndices[0]])

    #find next event(s); repeat until convergence
    localCounter = 0
    while True:
        localDF = df.iloc[np.where((tmpEvents[localCounter].Begin_Date <= df.Begin_Date) & (tmpEvents[localCounter].End_Date >= df.Begin_Date))]
        #sort to ensure that previously identified event is in first row
        localDF = localDF.sort_values(by=['Begin_Date','Total_Over_Extreme'], ascending=[True,False])
        output = findEvent(localDF)
        tmpEvents.append(output[0])
        tmpIndices.append(output[1])
        tmpDFs.append(localDF.iloc[tmpIndices[localCounter+1]])
        if tmpEvents[localCounter].equals(tmpEvents[localCounter+1]):
            break
        localCounter += 1

    allEvents = pd.concat(tmpDFs)
    allEvents.drop_duplicates(inplace=True)
    events.append(tmpEvents[localCounter])
    df.drop(allEvents.index.tolist(), inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(len(df))


events = pd.DataFrame(events)
events = events.sort_values(by='Begin_Date')
events.drop_duplicates(inplace=True)
events.reset_index(drop=True, inplace=True)
polys = [shapely.wkt.loads(i) for i in events.geometry]
events.to_csv('/share/data1/ty/database/database_v1.0.csv', index=False)

events = events.astype({'Begin_Date':str, 'End_Date':str})
gdf = geopandas.GeoDataFrame(events, geometry=polys)
gdf.to_file('/share/data1/ty/database/database_v1.0.shp')

import numpy as np
from netCDF4 import MFDataset, num2date
import datetime
import pandas as pd
import glob
import xesmf as xe
import argparse

def _getFiles(source):
    options = ['PRISM','Livneh']
    files = glob.glob(f'/scratch/tdickinson/{source}/*.nc')
    files = [i for i in files if 'ltm' not in i]
    files.sort()
    return files

def _loadData(begin, length, source):
    files = _getFiles(source=source)

    firstYr = int(files[0].split('.')[-2])
    lastYr = int(files[-1].split('.')[-2])

    nc = MFDataset(files, 'r')
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]

    times = nc.variables['time'][:]
    timeUnits = nc.variables['time'].units
    try: timeCalendar = nc.variables['time'].calendar
    except: timeCalendar = 'standard'
    times = num2date(times, timeUnits, timeCalendar)

    #generate a list of dates in the window to slice out of all times in the nc file
    window = []
    for i in range(firstYr, lastYr+1, 1):
        beginWindow = datetime.datetime(month=begin.month, day=begin.day, year=i)
        endWindow = beginWindow + datetime.timedelta(days=length-1)
        rng = pd.date_range(beginWindow, endWindow, freq='D')
        #remove leap days
        mask = (rng.month == 2) & (rng.day == 29)
        if any(mask):
            rng = rng[~mask].to_pydatetime()
            rng = np.append(rng, endWindow + datetime.timedelta(days=1))
            window.append(rng)
        else:
            window.append(rng.to_pydatetime())
    window = np.asarray(window).flatten()

    #remove excess days if window overlaps years
    if beginWindow.year != endWindow.year:
        window = window[:-length]

    #get indices of intersection in the times array
    _, iTime, _ = np.intersect1d(times, window, return_indices=True)
    data = nc.variables['prec'][iTime,:,:].squeeze()
    nc.close()
    return lat, lon, data

def loadWindowPrecip(begin, length, mode):
    livnehLat, livnehLon, livnehData = _loadData(begin=begin, length=length, source='Livneh')
    prismLat, prismLon, prismData = _loadData(begin=begin, length=length, source='PRISM')

    #bilinearly interpolate PRISM data to Livneh dataset grid
    gridIn = {'lat':prismLat, 'lon':prismLon+360}
    gridOut = {'lat':livnehLat, 'lon':livnehLon}
    regridder = xe.Regridder(gridIn, gridOut, method='bilinear', reuse_weights=True)
    prismRegrid = regridder(prismData)
    #create mask based on pre-existing NaN and where Livneh data extends further north
    mask = np.isnan(prismRegrid[0]) | (livnehLat > prismLat.max())[:,None]
    prismRegrid = np.ma.array(prismRegrid, mask=np.tile(mask, (prismRegrid.shape[0],1,1)))

    allData = np.ma.vstack([livnehData, prismRegrid])
    #mask rows that have any masked values in them
    outsideUS = allData.mask.any(axis=0)
    allData[:,outsideUS] = np.ma.masked

    #find window sum as a function of year if input is totals; find window mean otherwise
    if mode == 'totals':
        t,y,x = allData.shape
        totalYears = int(t/length)
        allData = np.nansum(allData.reshape(totalYears, length, y, x), axis=1)
    else:
        allData = np.nanmean(allData, axis=0)
    return allData


begin = datetime.datetime(month=11, day=30, year=1915)

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, help="number of days in window")
parser.add_argument("--mode", type=str, help="type of file to generate. options are totals and means. total is used to calculate window percentile whereas mean is used for window duration criteria.")
args = parser.parse_args()

length = args.length
mode = args.mode
modeOptions = ['totals', 'means']
if mode not in modeOptions:
    raise ValueError(f'{mode} is not a supported mode. Options are {modeOptions}')

for i in range(32):
    date = begin + datetime.timedelta(days=i)
    precip = loadWindowPrecip(begin=date, length=length, mode=mode)
    precip = precip.filled(np.nan)
    np.save(f'/scratch/tdickinson/Livneh/windows/{length}/{mode}/precip.{str(date.month).zfill(2)}{str(date.day).zfill(2)}.npy', precip)

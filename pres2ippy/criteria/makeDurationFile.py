import numpy as np
from netCDF4 import Dataset, date2num
import datetime
import argparse

def netCDF4Writer(data, name, length):
    """
    Function to create a netCDF4 Classic file to store relevant info for extreme
    precipitation event identification.

    Inputs
    ------
    data : dict
        Dictionary holding latitudes, longitudes, and duration thresholds.
    name : str
        Name for the netCDF file. Path is assumed to be included.
    length : int
        Length of window. Used for file metadata.
    """

    dataset = Dataset(name,'w',format='NETCDF4_CLASSIC')

    #there are 3 dimensions to my data: latitudes, longtiudes, and times
    lats = dataset.createDimension('lat', data['lat'].size)
    lons = dataset.createDimension('lon', data['lon'].size)
    times = dataset.createDimension('time', data['duration'].shape[0])

    #here, I create the variables to be stored in the file
    lat = dataset.createVariable('lat', np.float64, ('lat',), zlib=True, shuffle=True, complevel=6, fill_value=None)
    lon = dataset.createVariable('lon', np.float64, ('lon',), zlib=True, shuffle=True, complevel=6, fill_value=None)
    time = dataset.createVariable('time', np.float64, ('time',), zlib=True, shuffle=True, complevel=6, fill_value=None)
    duration = dataset.createVariable('duration', np.float64, ('time','lat','lon'), zlib=True, shuffle=True, complevel=6, fill_value=None)

    #variable attributes
    lat.standard_name = 'latitude'
    lat.units = 'degree_north'
    minimum = np.nanmin(data['lat'])
    maximum = np.nanmax(data['lat'])
    lat.actual_range = np.array([minimum,maximum])

    lon.standard_name = 'longitude'
    lon.units = 'degree_east'
    minimum = np.nanmin(data['lon'])
    maximum = np.nanmax(data['lon'])
    lon.actual_range = np.array([minimum,maximum])

    time.standard_name = 'time'
    time.calendar = 'standard'
    time.units = 'days since 1915-01-01 00:00:00'
    tmp = np.arange(data['duration'].shape[0])
    time.actual_range = np.array([tmp.min(), tmp.max()])

    duration.standard_name = 'duration'
    duration.long_name = f'duration threshold for {length}-day extreme precipitation events'
    duration.units = 'mm'

    dataset.description = 'File containing daily precipitation thresholds for the duration criteria across the CONUS for each window of length {length} of the year. Thresholds were calculated using the mean daily precipitation for all days in the window. Original data source was daily Livneh data plus interpolated daily PRISM data post 2011. The year attribute in the time object is arbitrary.'
    today = datetime.datetime.today()
    dataset.history = 'Created %d/%d/%d'%(today.month, today.day, today.year)
    dataset.source = 'Ty A. Dickinson'

    #store data in the variables created earlier
    lat[:] = data['lat']
    lon[:] = data['lon']

    dates = []
    for n in range(data['duration'].shape[0]):
        dates.append(datetime.datetime(year=1915, month=1, day=1) + n*datetime.timedelta(days=1))
    time[:] = date2num(dates,units=time.units,calendar=time.calendar)

    duration[:] = data['duration']

    dataset.close()
    return

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--length", type=int, help="number of days in window")
args = parser.parse_args()

length = args.length
begin = datetime.datetime(month=1, day=1, year=1915)
data = {}

with Dataset('/scratch/tdickinson/Livneh/prec.1915.nc','r') as nc:
    data['lat'] = nc.variables['lat'][:]
    data['lon'] = nc.variables['lon'][:]

numLats = data['lat'].size
numLons = data['lon'].size

data['duration'] = np.zeros((365, numLats, numLons))*np.nan
for i in range(data['duration'].shape[0]):
    date = begin + datetime.timedelta(days=i)
    data['duration'][i,:,:] = np.load(f'/scratch/tdickinson/Livneh/windows/{length}/means/precip.{str(date.month).zfill(2)}{str(date.day).zfill(2)}.npy')

fileName = f'/home/tdickinson/data/livnehPRISM.duration.n{length}.nc'
netCDF4Writer(data=data, name=fileName, length=length)

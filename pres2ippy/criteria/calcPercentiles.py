import numpy as np
from netCDF4 import Dataset, date2num
import datetime
import argparse

def getPercentiles(date, length, q):
    """
    Calculate a given percentile for precipitation data.

    Parameters
    ----------
    date : datetime obj.
        Begin date of window to load
    length : int
        Length of window. Ensures the correct file is loaded.
    q : int
        Percentile to calculate.

    Returns
    -------
    percs : array, shape (y,x)
        qth percentile for precipitation for the length-day window beginning on
        date. Final shape corresponds to the number of latitudes and number of longitudes.
    """
    #backfill month and day with zeros if necessary (e.g., 1 --> 01)
    month = str(date.month).zfill(2)
    day = str(date.day).zfill(2)
    path = f'/scratch/tdickinson/Livneh/windows/{length}/totals/precip.{month}{day}.npy'
    data = np.load(path)

    percs = np.nanpercentile(data, q=q, axis=0)
    return percs

def fourierSeries(period, N):
    """
    Calculate the Fourier series coefficients up to the Nth harmonic for a 2D
    array.

    Parameters
    ----------
    period : arr-like, shape (t, s)
        2D array with raw percentiles for each window throughout the calendar year.
        The first dimension should be time (i.e., t=365) and the second dimension
        should be the number of grid points in space, s.
    N : int
        Number of harmonics to calculate.

    Returns
    -------
    harmonics : array, shape (N+1, 2, s)
        3D array of Fourier harmonics. Shape corresponds to first N harmonics,
        both a and b, for all points in the grid.
    """
    harmonics = np.zeros((N+1,2,period.shape[1]))
    T = period.shape[0]
    t = np.arange(T)
    for n in range(N+1):
        an = 2/T*(period * np.cos(2*np.pi*n*t/T)[:,np.newaxis]).sum(axis=0)
        bn = 2/T*(period * np.sin(2*np.pi*n*t/T)[:,np.newaxis]).sum(axis=0)
        harmonics[n,0,:] = an
        harmonics[n,1,:] = bn
    return harmonics

def reconstruct(P, anbn):
    """
    Reconstruct the signal using a reduced number of harmonics.

    Parameters
    ----------
    P : int
        Length of signal to reconstruct. To reconstruct for the entire year, set P=365.
    anbn : arr-like, shape (N, 2, s)
        a and b Fourier coefficients to reconstruct time series. Use output from
        fourierSeries function.

    Returns
    -------
    result : array, shape (P, s)
        Reconstructed time series using the reduced number of Fourier harmonics.
    """
    result = np.zeros((P, anbn.shape[-1]))
    t = np.arange(P)
    for n, (a, b) in enumerate(anbn):
        if n == 0:
            a /= 2
        aTerm = a[np.newaxis,:] * np.cos(2*np.pi*n*t/P)[:,np.newaxis]
        bTerm = b[np.newaxis,:] * np.sin(2*np.pi*n*t/P)[:,np.newaxis]
        result += aTerm + bTerm
    return result

def consecutive(data, stepsize=1):
    """
    Helper function for fixNegativeThresholds to group consecutive elements
    in a NumPy array.

    Parameters
    ----------
    data : arr-like
        NumPy array containing indices with negative percentile thresholds.
    stepsize : int, default = 1
        Size between elements to be grouped together. Setting stepsize=1 (default)
        groups consecutive elements.

    Returns
    -------
    Array of arrays where each array contains consecutive elements.
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def fixNegativeThresholds(raw, smoothed):
    """
    Function to remove unphysical, negative precipitation thresholds that arise
    from smoothing via a reduced number of Fourier harmonics. Negative values are
    replaced with the mean values over the same indices (i.e., time windows) before
    smoothing occurred.

    Parameters
    ----------
    raw : arr-like
        Raw precipitation thresholds before Fourier smoothing. In other words,
        the window-by-window percentiles.
    smoothed : arr-like
        Precipitation thresholds after Fourier smoothing.

    Returns
    -------
    new : numpy.ndarray
        Copy of smoothed except with negative indices replaced with the mean raw
        values over the same indices.
    """
    new = smoothed.copy()
    for i in range(new.shape[-1]):
        negLocs = np.where(smoothed[:,i] <= 0)[0]
        if all(np.isnan(smoothed[:,i])) or (negLocs.size == 0):
            continue
        else:
            negLocs = consecutive(negLocs)
            for j in negLocs:
                new[j,i] = np.mean(raw[j,i])
    return new

def netCDF4Writer(data, name, length, q, n):
    """
    Function to create a netCDF4 Classic file to store relevant info for extreme
    precipitation event identification.

    Inputs
    ------
    data : dict
        Dictionary holding latitudes, longitudes, and percentile thresholds.
    name : str
        Name for the netCDF file. Path is assumed to be included.
    length : int
        Length of window. Used for file metadata.
    q : int
        Percentile used to form thresholds. Used for file metadata.
    n : int
        Number of retained Fourier harmonics in smoothing. Used for file metadata.
    """

    dataset = Dataset(name,'w',format='NETCDF4_CLASSIC')

    #there are 3 dimensions to my data: latitudes, longtiudes, and times
    lats = dataset.createDimension('lat', data['lat'].size)
    lons = dataset.createDimension('lon', data['lon'].size)
    times = dataset.createDimension('time', data['threshold'].shape[0])

    #here, I create the variables to be stored in the file
    lat = dataset.createVariable('lat', np.float64, ('lat',), zlib=True, shuffle=True, complevel=6, fill_value=None)
    lon = dataset.createVariable('lon', np.float64, ('lon',), zlib=True, shuffle=True, complevel=6, fill_value=None)
    time = dataset.createVariable('time', np.float64, ('time',), zlib=True, shuffle=True, complevel=6, fill_value=None)
    threshold = dataset.createVariable('threshold', np.float64, ('time','lat','lon'), zlib=True, shuffle=True, complevel=6, fill_value=None)

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

    threshold.standard_name = 'threshold'
    threshold.long_name = f'threshold for {length}-day extreme precipitation events (percentile={q})'
    threshold.units = 'mm'

    dataset.description = f'File containing precipitation thresholds for the CONUS for each window of length {length} of the year. Thresholds were calculated using the {q} percentile and smoothed using the first {n} Fourier harmonics. Original data source was daily Livneh data plus interpolated daily PRISM data post 2011. The year attribute in the time object is arbitrary.'
    today = datetime.datetime.today()
    dataset.history = 'Created %d/%d/%d'%(today.month, today.day, today.year)

    #store data in the variables created earlier
    lat[:] = data['lat']
    lon[:] = data['lon']

    dates = []
    for n in range(data['threshold'].shape[0]):
        dates.append(datetime.datetime(year=1915, month=1, day=1) + n*datetime.timedelta(days=1))
    time[:] = date2num(dates,units=time.units,calendar=time.calendar)

    threshold[:] = data['threshold']

    dataset.close()
    return

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--length", type=int, help="number of days in window")
parser.add_argument("-p", "--percentile", type=float, help="percentile to calculate")
parser.add_argument("-c", "--components", type=int, help="number of Fourier harmonics to keep")
args = parser.parse_args()

length = args.length
percentile = args.percentile
numComponents = args.components
begin = datetime.datetime(month=1, day=1, year=1915)
data = {}

with Dataset('/scratch/tdickinson/Livneh/prec.1915.nc','r') as nc:
    data['lat'] = nc.variables['lat'][:]
    data['lon'] = nc.variables['lon'][:]

numLats = data['lat'].size
numLons = data['lon'].size

percs = np.zeros((365, numLats, numLons))*np.nan
for i in range(percs.shape[0]):
    date = begin + datetime.timedelta(days=i)
    print(date)
    percs[i,:,:] = getPercentiles(date=date, length=length, q=percentile)

percs = percs.reshape(365, numLats*numLons)
coefs = fourierSeries(period=percs, N=numComponents)
threshold = reconstruct(P=365, anbn=coefs)
data['threshold'] = fixNegativeThresholds(raw=percs, smoothed=threshold)
data['threshold'] = data['threshold'].reshape(365, numLats, numLons)

fileName = f'/scratch/tdickinson/files/livnehPRISM.thresholds.q{int(percentile)}.n{length}.nc'
netCDF4Writer(data=data, name=fileName, length=length, q=percentile, n=numComponents)

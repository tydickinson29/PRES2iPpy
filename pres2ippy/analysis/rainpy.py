import helpers
import numpy as np
from netCDF4 import MFDataset,Dataset,num2date
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from shapely.geometry import Polygon
import pandas as pd
import datetime
import os
import warnings
#import time as t
warnings.simplefilter("ignore")

"""
try:
    import salem
    import xarray as xr
    _applyMask = True
except ImportError as e:
    print('%s, will not be able to mask precip obs outside of CONUS.'%e)
    _applyMask = False
"""

class DateTest(object):
    """Test an input date between January 1, 1915 and December 31, 2018 to see
    if an S2S extreme event occurred.

    Parameters
    ----------
    month : int
        The month of the event.
    day : int
        The day of the event.
    year : int
        The year of the event. Currently must be between 1915 and 2011, inclusive.
    length : int
        Length of model to be used in analysis. Current option is only 14; 30, 60, and 90 to be added.

    Attributes
    ----------
    DATE_BEGIN: datetime
        Datetime object specifying the beginning date.
    DATE_END : datetime
        Datetime object specifying the ending date.
    lat : array, shape (444)
        Latitudes for the quantile regression model grid.
    lon : array, shape (922)
        Longitudes for the quantile regression model grid.
    time : array, shape (length)
        Datetime objects specifying all days being analyzed.
    intercept : array, shape (lat, lon)
        y-intercepts of the quantile regression model.
    slope : array, shape (lat, lon)
        Slopes of the quantile regression model.
    model : array, shape (lat, lon)
        95th percentile grid; calculated by doing ``intercept`` + ``slope`` x ``year`` for the input ``day``.
    kdeGridX : array, shape (261, 622)
        Longitude grid the kernel density estimation is evaluated onto. Range is [128W, 66W] every 1/10th of a degree.
    kdeGridY : array, shape (261, 622)
        Latitude grid the kernel density estimation is evaluated onto. Range is [24N, 50N] every 1/10th of a degree.
    obs : array, shape (length, lat, lon)
        Recorded precipitation each day from the Livneh dataset for the input 14-day period. Filled after :func:`getObs` is called.
    units : string
        Units of any precipitation measurement or statistic unless otherwise noted. Filled after :func:`getObs` is called.
    total : array, shape(lat, lon)
        Total precipitation from the Livneh dataset (i.e., sum at each grid point of ``obs``). Filled after :func:`getObs` is called.
    diff : array, shape (lat, lon)
        Difference between ``obs`` and ``model``. Filled after :func:`getObs` is called.
    means : array, shape (lat, lon)
        Daily mean precipitation for window being examined; calculated using Livneh period of record. Filled after :func:`checkDuration` is called.
    duration : array, shape (lat, lon)
        Number of days in the window being examined that meet or exceed the daily mean precipitation threshold. Filled after :func:`checkDuration` is called.
    extreme : array, shape (lat, lon)
        True where ``diff`` is positive and ``daysOver2`` is at least 5; False if either condition is not met. Filled after :func:`getExtremePoints` is called.
    KDE: object
        Parameters being used in scikit-learn's `KernelDensity <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity>`_ class.
    density : array, shape (kdeGridX, kdeGridY)
        Density (height in the vertical coordinate) obtained from the KDE analysis. Filled after :func:`kde` is called.
    polys : list
        List of Shapely `Polygon <https://shapely.readthedocs.io/en/stable/manual.html#polygons>`_ s describing the extreme region(s). Filled after :func:`getAreas` is called.
    areas : dict
        Areas of KDE (:func:`kde`) contours drawn in square kilometers. Filled after :func:`getAreas` is called.
    regionsDaily : array, shape (numRegions, length*lat*lon)
        Daily precipitation with all areas outside each region set to NaN.
    regionsTotal : array, shape (numRegions, lat*lon)
        Event total precipitation with all areas outside each region set to NaN.
    regionsDiff : array, shape (numRegions)
        Event totals over extreme threshold for all positive points inside region.
    weightedTotal : array, shape (numRegions)
        Areal-averaged precipitation for all points inside extreme region.
    numPoints : int
        Number of grid points inside each bounded KDE region.
    """

    #global _applyMask
    _daysInMonth = {'1':['January',31], '2':['February',28], '3':['March',31],
                    '4':['April',30], '5':['May',31], '6':['June',30],
                    '7':['July',31], '8':['August',31], '9':['September',30],
                    '10':['October',31], '11':['November',30], '12':['December',31]}

    def __init__(self,month,day,year,length=14,dataset='livneh'):
        self.month = month
        self.day = day
        self.year = year
        self.length = length
        self.dataset = dataset

        #FIXME: Down the road, will have 1 netCDF4 file for all 95 percentiles
        #with lengths 14, 30, 60, and 90. Take additional argument in constructor
        #to slice intercept, slope for correct length
        with Dataset('/scratch/tdickinson/%s.quantReg.95.14.nc'%(self.dataset),'r') as nc:
            self.lat = nc.variables['lat'][:]
            self.lon = nc.variables['lon'][:]
            #length = nc.variables['length'][:]
            time = nc.variables['time'][:]
            timeUnits = nc.variables['time'].units
            timeCalendar = nc.variables['time'].calendar
            self.time = num2date(time,timeUnits,timeCalendar)
            self._month = np.array([d.month for d in self.time])
            self._day = np.array([d.day for d in self.time])

            #ilength = np.where(length == self.length)[0][0]
            itime = np.where((self._month == self.month) & (self._day == self.day))[0][0]

            self.intercept = nc.variables['intercept'][itime,:,:]
            self.slope = nc.variables['slope'][itime,:,:]
            del time,timeUnits,timeCalendar,nc

        self.model = self.intercept + self.slope*self.year

        gridLats = np.arange(18, 62.1, 0.1)
        gridLons = np.arange(228, 302.1, 0.1)
        self.kdeGridX, self.kdeGridY = np.meshgrid(gridLons, gridLats)

        self.obs = None
        self.units = ''
        self.total = None
        self.diff = None
        self.means = None
        self.duration = None
        self.extreme = None
        self.density = None
        self.areas = {}
        self.polys = []
        self._noPoints = False
        self.regionsDaily = None
        self.regionsTotal = None
        self.regionsDiff = None
        self.weightedTotal = None
        self.numPoints = None

    def __repr__(self):
        return 'DateTest(month=%s, day=%s, year=%s)'%(self.month, self.day, self.year)

    @property
    def month(self):
        """Get or set the current month. Setting the month to a new value will
        set the ``DATE_BEGIN`` and ``DATE_END`` properties automatically.
        """
        return self.__month

    @month.setter
    def month(self,val):
        if str(val) not in self._daysInMonth.keys():
            raise ValueError('%s is not a valid month.'%val)
        else:
            self.__month = val

    @property
    def day(self):
        """Get or set the current day. Setting the day to a new value will
        set the ``DATE_BEGIN`` and ``DATE_END`` properties automatically.
        """
        return self.__day

    @day.setter
    def day(self,val):
        monInfo = self._daysInMonth[str(self.month)]
        if (val > monInfo[1]) or (val <= 0):
            raise ValueError('%s is not a valid day in %s.'%(val,monInfo[0]))
        else:
            self.__day = val

    @property
    def year(self):
        """Get or set the current year. Setting the year to a new value will
        set the ``DATE_BEGIN`` and ``DATE_END`` properties automatically.
        """
        return self.__year

    @year.setter
    def year(self,val):
        if (val < 1915):
            raise ValueError('Years before 1915 are currently not supported.')
        elif (val > 2018):
            raise ValueError('Years after 2018 are currently not supported.')
        else:
            self.__year = val

    @property
    def length(self):
        """Get or set the length of the window being considered.
        """
        return self.__length

    @length.setter
    def length(self,val):
        validLengths = [14]
        if val not in validLengths:
            raise ValueError('%s is currently not a supported length.'%(val))
        else:
            self.__length = val

    @property
    def dataset(self):
        """Get or set the dataset to use.
        """
        return self.__dataset

    @dataset.setter
    def dataset(self,val):
        validDatasets = ['livneh','era5']
        if val not in validDatasets:
            raise ValueError('%s is currently not a supported dataset.'%(val))
        else:
            self.__dataset = val

    @property
    def DATE_BEGIN(self):
        """Set the beginning date instance attribute. ``DATE_BEGIN`` will be type `datetime <https://docs.python.org/2/library/datetime.html>`_.
        """
        return datetime.datetime(self.year, self.month, self.day)

    @property
    def DATE_END(self):
        """Set the ending date instance attribute. ``DATE_END`` will be type `datetime <https://docs.python.org/2/library/datetime.html>`_.
        ``DATE_BEGIN`` is incremented by 13 days to have an inclusive 14-day window.
        """
        return self.DATE_BEGIN + datetime.timedelta(days=13)
        #return self.DATE_BEGIN + datetime.timedelta(days=self.length-1)

    @property
    def datasetBegin(self):
        if self.dataset == 'era5':
            return 'ERA5'
        else:
            if self.DATE_BEGIN.year < 2012:
                return 'Livneh'
            else:
                return 'PRISM'

    @property
    def datasetEnd(self):
        if self.dataset == 'era5':
            return 'ERA5'
        else:
            if self.DATE_END.year < 2012:
                return 'Livneh'
            else:
                return 'PRISM'

    @property
    def isLeapDay(self):
        if (self.month == 2) and (self.day == 29):
            raise ValueError('Leap day is currently an unsupported input date.')

    """
    def _mask(self, arrToMask):
        arr = xr.DataArray(arrToMask, dims=['lat','lon'],
                coords={'lat':self.lat,'lon':self.lon})

        shp = salem.read_shapefile('/share/data1/Students/ty/NA_shapefile/North_America.shp')
        shpSlice = shp.loc[shp['NAME'].isin(['UNITED STATES'])]
        test = arr.salem.roi(shape=shpSlice)
        test = test.values
        return test
    """

    def prepInterp(self, data, split=False, firstPiece=None):
        """Interpolate PRISM grid to Livneh grid in order to correctly compare to extreme thresholds.

        Parameters
        ----------
        data : array
            Array to be interpolated
        split : boolean
            If False (default), the time period does not begin in 2011 and end in 2012 (crossover from Livneh to PRISM).
        firstPiece : NoneType or array
            Ignore if split is False. If split is True, an array should be given holding Livneh data.
        """

        with Dataset('/scratch/tdickinson/PRISM/prec.2018.nc','r') as nc:
            latPRISM = nc.variables['lat'][:]
            lonPRISM = nc.variables['lon'][:]

        with Dataset('/scratch/tdickinson/Livneh/prec.1915.nc', 'r') as nc:
            latLivneh = nc.variables['lat'][:]
            lonLivneh = nc.variables['lon'][:] - 360.

        self._iX = np.where((lonLivneh >= lonPRISM.min()) & (lonLivneh <= lonPRISM.max()))[0]
        self._iY = np.where((latLivneh >= latPRISM.min()) & (latLivneh <= latPRISM.max()))[0]
        #Adjust lats and lons to new interpolated grid (N-S bounds slightly less)
        self.lat = latLivneh[self._iY]
        self.lon = lonLivneh[self._iX]
        lonMesh,latMesh = np.meshgrid(lonLivneh[self._iX],latLivneh[self._iY])

        #interpolate PRISM obs to Livneh grid each day at a time
        tmp = np.zeros((data.shape[0],lonMesh.shape[0],lonMesh.shape[1]))
        for i in range(tmp.shape[0]):
            tmp[i,:,:] = helpers.interp(datain=data[i,:,:], xin=lonPRISM, yin=latPRISM[::-1],
                            xout=lonMesh, yout=latMesh)
        if split:
            self.obs = np.concatenate((firstPiece[:,self._iY,:][:,:,self._iX], tmp), axis=0)
        else:
            self.obs = tmp
        #adjust model to new grid
        self.model = self.model[self._iY,:][:,self._iX]
        return

    def getObs(self):
        """Retrive Livneh or PRISM data from the year specified by the object.

        Creates the observations and difference attributes for the instance. Observations
        are from Livneh and the difference is the observation amounts minus the amount given
        by the quantile regression model. Furthermore, the differences are specified to be 1
        if the rainfall was greater than the extreme threshold and 0 if less than the extreme
        threshold.
        """
        split = False
        tag = 'prec'
        if self.dataset == 'era5':
            beginPath = endPath = '/scratch/tdickinson/era5/tp_daily_'
            tag = 'tp'
            latBounds = [20,60]
            lonBounds = [230,300]
            with Dataset('%s1979.nc'%(beginPath),'r') as nc:
                lats = nc.variables['latitude'][:]
                self._iLat = np.where((lats >= latBounds[0]) & (lats <= latBounds[1]))[0]
                lons = nc.variables['longitude'][:]
                self._iLon = np.where((lons >= lonBounds[0]) & (lons <= lonBounds[1]))[0]
        else:
            if (self.datasetBegin == self.datasetEnd) and (self.datasetBegin == 'Livneh'):
                beginPath = endPath = '/scratch/tdickinson/Livneh/prec.'
            elif (self.datasetBegin == self.datasetEnd) and (self.datasetBegin == 'PRISM'):
                beginPath = endPath = '/scratch/tdickinson/PRISM/prec.'
            else:
                beginPath = '/scratch/tdickinson/Livneh/prec.'
                endPath = '/scratch/tdickinson/PRISM/prec.'
                split = True

        if self.DATE_BEGIN.year == self.DATE_END.year:
            with Dataset(beginPath+'%d.nc'%(self.year),'r') as nc:
                #print('Getting observations from %s'%self.year)
                time = nc.variables['time'][:]
                timeUnits = nc.variables['time'].units
                timeCalendar = nc.variables['time'].calendar
                time = num2date(time,timeUnits,timeCalendar)
                month = np.array([d.month for d in time])
                day = np.array([d.day for d in time])

                if self.DATE_BEGIN.month == self.DATE_END.month:
                    self._locs = np.where(((month==self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) & ((month==self.DATE_END.month)&(day<=self.DATE_END.day)))[0]
                else:
                    self._locs = np.where(((month==self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) | ((month==self.DATE_END.month)&(day<=self.DATE_END.day)))[0]

                if self.dataset == 'era5':
                    self.obs = nc.variables[tag][self._locs,self._iLat,self._iLon]
                else:
                    self.obs = nc.variables[tag][self._locs,:,:]
                self.units = nc.variables[tag].units

            self.obs = self.obs.filled(np.nan)
            self._time = time[self._locs]
            if self.datasetBegin == 'PRISM':
                self.prepInterp(data=self.obs[:,::-1,:])

        else:
            #here, the window goes into the following year so we must load two files
            with Dataset(beginPath+'%d.nc'%(self.DATE_BEGIN.year), 'r') as nc:
                time = nc.variables['time'][:]
                timeUnits = nc.variables['time'].units
                timeCalendar = nc.variables['time'].calendar
                time1 = num2date(time,timeUnits,timeCalendar)
                month = np.array([d.month for d in time1])
                day = np.array([d.day for d in time1])

                time1Locs = np.where(((month>=self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) & ((month<=12)&(day<=31)))[0]
                if self.dataset == 'era5':
                    time1Obs = nc.variables[tag][time1Locs,self._iLat,self._iLon]
                else:
                    time1Obs = nc.variables[tag][time1Locs,:,:]
                time1Obs = time1Obs.filled(np.nan)
                self.units = nc.variables[tag].units

            with Dataset(endPath+'%d.nc'%(self.DATE_END.year), 'r') as nc:
                time = nc.variables['time'][:]
                timeUnits = nc.variables['time'].units
                timeCalendar = nc.variables['time'].calendar
                time2 = num2date(time,timeUnits,timeCalendar)
                month = np.array([d.month for d in time2])
                day = np.array([d.day for d in time2])

                time2Locs = np.where(((month>=1)&(day>=1)) & ((month<=self.DATE_END.month)&(day<=self.DATE_END.day)))[0]
                if self.dataset == 'era5':
                    time2Obs = nc.variables[tag][time2Locs,self._iLat,self._iLon]
                else:
                    time2Obs = nc.variables[tag][time2Locs,:,:]
                time2Obs = time2Obs.filled(np.nan)

            if ((not split) and (self.datasetEnd == 'Livneh')) or (self.dataset == 'era5'):
                self.obs = np.concatenate((time1Obs, time2Obs), axis=0)
            elif (not split) and (self.datasetBegin == 'PRISM'):
                self.obs = np.concatenate((time1Obs, time2Obs), axis=0)
                self.prepInterp(data=self.obs[:,::-1,:])
            else:
                self.prepInterp(data=time2Obs[:,::-1,:], split=True, firstPiece=time1Obs)

            self._locs = np.concatenate((time1Locs,time2Locs), axis=None)
            self._time = np.concatenate((time1[time1Locs],time2[time2Locs]), axis=None)

        if self.units == 'm':
            self.obs *= 1000
            self.units = 'mm'
        self.total = np.sum(self.obs,axis=0)
        #if self.datasetEnd == 'Livneh':
            #self.model = self._mask(self.model)
            #self.total = self._mask(self.total)
        self.diff = self.total - self.model
        return

    def checkDuration(self):
        """Check if at least half of the period experienced above normal daily precipitation.
        Normal daily precipitation is defined as the mean of all days in the period in the Livneh
        period of record (1915-2011).
        """
        if self.diff is None:
            self.getObs()

        if self.dataset == 'era5':
            path = '/scratch/tdickinson/era5/tp_daily_ltm.nc'
        else:
            path = ''

        with Dataset(path,'r') as nc:
            """
            time = nc.variables['time'][:]
            timeUnits = nc.variables['time'].units
            timeCalendar = nc.variables['time'].calendar
            time = num2date(time,timeUnits,timeCalendar)
            months = np.array([d.month for d in time])
            days = np.array([d.day for d in time])
            if self.DATE_BEGIN.month == self.DATE_END.month:
                iloc = np.where(((months==self.DATE_BEGIN.month)&(days>=self.DATE_BEGIN.day)) & ((months==self.DATE_END.month)&(days<=self.DATE_END.day)))[0]
            else:
                iloc = np.where(((months==self.DATE_BEGIN.month)&(days>=self.DATE_BEGIN.day)) | ((months==self.DATE_END.month)&(days<=self.DATE_END.day)))[0]
            leapDay = np.where((months==2) & (days==29))[0][0]
            idx = np.where(iloc == leapDay)[0]
            if idx.size != 0:
                iloc = np.delete(iloc,idx)
            """
            if self.dataset == 'era5':
                self.means = nc.variables['tp'][self._locs,self._iLat,self._iLon] * 1000
            else:
                self.means = nc.variables['prec'][self._locs,:,:]

        t,y,x = self.obs.shape
        tmpObs = self.obs.reshape(t,y*x)
        tmpMeans = self.means.reshape(t,y*x)
        nonNaN = np.where(~np.isnan(tmpObs[0,:]))[0]
        self.duration = np.zeros(y*x)*np.nan
        for i in nonNaN:
            diff = tmpObs[:,i] - tmpMeans[:,i]
            self.duration[i] = len(np.where(diff > 0)[0])
        self.duration = self.duration.reshape(y,x)
        return

    def getExtremePoints(self):
        """Find which points are extreme.

        Points must have exceeded the 14-day 95th percentile and experienced at
        least 7 days of above normal daily precipitation.
        """
        if self.diff is None:
            self.getObs()
        if self.duration is None:
            self.checkDuration()

        self.extreme = (self.diff >= 0) & (self.duration >= (self.length / 2.))
        return

    def kde(self, weighted=False, **kwargs):
        """Calculate the kernel density estimate for a given period.

        Additional keyword arguments are accepted to customize the KernelDensity class.
        Every third Livneh grid point is used; thus, the KDE grid is every 3/16 of a
        degree. Z is assigned as a public attribute and is the result of the
        KDE analysis.

        Default arguments passed to the KernelDensity class are the haversine distance metric,
        the epanechnikov kernel with 0.02 bandwidth, and the ball_tree algorithm.

        Parameters
        ----------
        weighted : boolean or None
            If True, weight the KDE fit based on magnitude over the extreme threshold.
            If None, do not assign weights.
        **kwargs
            Additional keyword arguments to scikit-learn's `KernelDensity
            <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity>`_
            class.
        """
        if (type(weighted) != type(True)) and (type(weighted) != type(None)):
            raise TypeError('weighted must be a bool or NoneType argument.')
        elif not weighted:
            weighted = None

        if self.extreme is None:
            self.getExtremePoints()

        kwargs.setdefault('bandwidth', 0.02)
        kwargs.setdefault('metric', 'haversine')
        kwargs.setdefault('kernel', 'epanechnikov')
        kwargs.setdefault('algorithm', 'ball_tree')

        x,y = np.meshgrid(self.lon, self.lat)
        locs = np.ma.where(self.extreme == 1)
        if locs[0].size == 0:
            #print('No extreme points')
            self._noPoints = True
            self.density = np.zeros_like(self.kdeGridX)
            return
        Xtrain = np.zeros((locs[0].size, 2)) * np.nan

        if weighted:
            weighted = np.zeros((locs[0].size)) * np.nan
            for i in range(Xtrain.shape[0]):
                Xtrain[i,0] = y[locs[0][i], locs[1][i]]
                Xtrain[i,1] = x[locs[0][i], locs[1][i]]
                weighted[i] = self.diff[locs[0][i],locs[1][i]]
            #divide by total so sum of weights is 1
            weighted /= np.sum(weighted)
        else:
            for i in range(Xtrain.shape[0]):
                Xtrain[i,0] = y[locs[0][i], locs[1][i]]
                Xtrain[i,1] = x[locs[0][i], locs[1][i]]

        #convert from lat/lon to radians
        self._XtrainRad = Xtrain * np.pi / 180.
        xy = np.vstack((self.kdeGridY.ravel(), self.kdeGridX.ravel())).T
        xy *= np.pi / 180.

        self.KDE = KernelDensity(**kwargs)
        self.KDE.fit(self._XtrainRad, sample_weight=weighted)
        self.density = np.exp(self.KDE.score_samples(xy))
        #divide by max to normalize density field
        self.density = self.density / np.nanmax(self.density)
        self.density = self.density.reshape(self.kdeGridX.shape)
        return

    def calcKDEPercentile(self, perc=95):
        """Return a KDE density for a given percentile.

        Parameters
        ----------
        perc : float in range of [0,100]
            Percentile to find in the KDE distribution; must be between 0 and 100, inclusive (defaults to 95)
        """
        if self._noPoints:
            return np.nan
        elif self.density is None:
            self.kde()
        return np.nanpercentile(a=self.density, q=perc)

    def getContours(self, ticks=None):
        """Method to draw contour for extreme region.

        Utilizing ``cartopy`` to setup Albers Equal Area projection since ``Basemap``
        is not installed in the environment being used to make the shapefile.
        """
        import cartopy.crs as ccrs

        if self._noPoints:
            return

        if ticks is None:
            levels = [self.calcKDEPercentile()]
        else:
            levels = [ticks]

        self._origProj = ccrs.PlateCarree()
        self._targetProj = ccrs.AlbersEqualArea(central_latitude=35, central_longitude=250)

        ax = plt.axes(projection=self._targetProj)
        ax.set_extent([228,302,18,62])
        self._im = plt.contour(self.kdeGridX, self.kdeGridY, self.density, levels=levels,
                            transform=self._origProj)

        self._levels = self._im.levels
        return

    def _coordTransform(self, x, y):
        return self._targetProj.transform_points(self._origProj, x, y)

    def calcAreas(self, areaThreshold):
        """Calculate the area of the polygons in the KDE map shown by :func:`getContours`.

        The vertices are gathered from the objected returned by matplotlib's
        `contour <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html>`_
        method and the area is calculated using Green's Theorem. Requires the :func:`getContours` method
        to have been called since it requires vertices to describe the polygon that we are
        finding the area of. Areas for each contour are stored in a dictionary and assigned as an
        instance attribute with units of squared kilometers.

        Parameters
        ----------
        areaThreshold : int or float
            Areal threshold to consider a region an extreme event
        """
        if self._noPoints:
            return

        try:
            getattr(self, '_im')
        except AttributeError:
            if self.density is None:
                self.kde()
            self.getContours()

        numContours = len(self._im.collections)
        for i in range(numContours):
            areas = []
            for region in self._im.collections[i].get_paths():
                trans = self._coordTransform(x=region.vertices[:,0], y=region.vertices[:,1])
                x = trans[:,0]
                y = trans[:,1]
                a = 0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
                a = np.abs(a) / (1000.**2)
                areas.append(a)
                if a >= areaThreshold:
                    self.polys.append(Polygon([(j[0], j[1]) for j in zip(region.vertices[:,0],region.vertices[:,1])]))
            self.areas[self._levels[i]] = areas

        #FIXME: fill Polygons with small holes inside
        return

    def maskOutsideRegion(self):
        """Set all points outside extreme region to NaN.

        Additionally, arrays are created to find areal-averaged precipitation,
        maximum 1-day and 14-day precipitation, the total precipitation over
        the extreme threshold in the extreme region, and the number of grid points
        inside the region.
        """
        import xarray as xr
        import salem

        if self._noPoints:
            self.density = np.zeros_like(self.kdeGridX)
            return
        elif len(self.polys) == 0:
            print('No extreme regions')
            return
        else:
            numRegions = len(self.polys)

            #first, make xarrays for daily and 14-day precipitation
            #add 360 since cartopy returns vertices in [0,360] and lons are all negative
            if self.dataset == 'livneh':
                lons = self.lon + 360.
            elif self.dataset == 'era5':
                lons = self.lon
            daily = xr.DataArray(self.obs, dims=('time', 'lat', 'lon'), coords={'time':self._time, 'lat':self.lat, 'lon':lons})
            total = xr.DataArray(self.total, dims=('lat', 'lon'), coords={'lat':self.lat, 'lon':lons})

            #define array where only points flagged as extreme are unmasked
            extremeDiffs = np.ma.masked_array(self.diff, ~self.extreme)
            diff = xr.DataArray(extremeDiffs, dims=('lat', 'lon'), coords={'lat':self.lat, 'lon':lons})

            #get array of ones with shape of kde grid and repeat numRegions times
            regions = np.ones_like(self.kdeGridX)
            regions = np.tile(regions, (numRegions,1)).reshape(numRegions,self.kdeGridX.shape[0],self.kdeGridX.shape[1])

            self.regionsDaily = np.zeros((len(self.polys), daily.shape[0], self.lat.size, self.lon.size)) * np.nan
            self.regionsTotal = np.zeros((len(self.polys), self.lat.size, self.lon.size)) * np.nan
            self.regionsDiff = np.zeros((len(self.polys), self.lat.size, self.lon.size)) * np.nan
            regions = xr.DataArray(regions, dims=('regions','lat','lon'), coords={'regions':range(numRegions),'lat':self.kdeGridY[:,0],'lon':self.kdeGridX[0,:]})
            regionsMasked = np.ones_like(regions)
            for i in range(numRegions):
                self.regionsDaily[i,:,:,:] = daily.salem.roi(geometry=self.polys[i])
                self.regionsTotal[i,:,:] = total.salem.roi(geometry=self.polys[i])
                self.regionsDiff[i,:,:] = diff.salem.roi(geometry=self.polys[i])
                regionsMasked[i,:,:] = regions[i,:,:].salem.roi(geometry=self.polys[i])

            #calc the cosine of the latitudes for the weights and average across the lons
            weights = np.cos(np.radians(self.lat))
            tmp = np.nanmean(self.regionsTotal, axis=2)
            #mask nans to accurately calculate weighted average
            tmp = np.ma.masked_array(tmp, np.isnan(tmp))
            #take average over lats
            self.weightedTotal = np.ma.average(tmp, axis=1, weights=weights)

            p,t,y,x = self.regionsDaily.shape
            self.regionsDaily = self.regionsDaily.reshape(p,t*y*x)
            p,y,x = self.regionsTotal.shape
            self.regionsTotal = self.regionsTotal.reshape(p,y*x)
            self.regionsDiff = self.regionsDiff.reshape(p,y*x)

            #find number of points inside each region
            self.numPoints = np.zeros((numRegions))*np.nan
            for i in range(numRegions):
                self.numPoints[i] = np.where(~np.isnan(regionsMasked[i,:,:]))[0].size
            """
            ax = plt.axes(projection=self._targetProj)
            ax.set_extent([232,294,24,50])
            im = plt.contourf(self.lon, self.lat, self.regions[0,:,:],
                                transform=self._origProj)
            plt.colorbar(im)
            ax.coastlines()
            plt.show(block=False)
            """
            return
    """
    def calcOptimalBandwidth(self,params):
        grid = GridSearchCV(KernelDensity(kernel='epanechnikov', metric='haversine'), params, n_jobs=10)
        grid.fit(self._XtrainRad)

        print('best bandwidth: %f'%grid.best_estimator_.bandwidth)
        return
    """

import numpy as np
from netCDF4 import Dataset,num2date
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from shapely.geometry import Polygon
import pandas as pd
import datetime
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
    """Make plots of potential 14-day extreme precipitation events and
    see their associated kernel density estimation maps to denote the extreme
    region.

    Parameters
    ----------
    month : int
        The month of the event.
    day : int
        The day of the event.
    year : int
        The year of the event. Currently must be between 1915 and 2011, inclusive.

    Attributes
    ----------
    lat : array, shape (444,)
        Latitudes for the quantile regression model grid.
    lon : array, shape (922)
        Longitudes for the quantile regression model grid.
    intercept : array, shape (lat, lon)
        y-intercepts of the quantile regression model.
    slope : array, shape (lat, lon)
        Slopes of the quantile regression model.
    length : int
        Number of days to consider.
    model : array, shape (lat, lon)
        95th percentile grid; calculated by doing ``intercept`` + ``slope`` x ``year`` for the input ``day``.
    kdeGridX : array, shape (261, 622)
        Longitude grid the kernel density estimation is evaluated onto. Range is [128W, 66W] every 1/10th of a degree.
    kdeGridY : array, shape (261, 622)
        Latitude grid the kernel density estimation is evaluated onto. Range is [24N, 50N] every 1/10th of a degree.
    obs : array, shape (lat, lon)
        Recorded precipitation each day from the Livneh dataset for the input 14-day period. Filled after :func:`getObs` is called.
    units : string
        Units of any precipitation measurement or statistic unless otherwise noted. Filled after :func:`getObs` is called.
    total : array, shape(lat, lon)
        Total precipitation from the Livneh dataset (i.e., sum at each grid point of ``obs``). Filled after :func:`getObs` is called.
    diff : array, shape (lat, lon)
        Difference between ``obs`` and ``model``. Filled after :func:`getObs` is called.
    daysOver2 : array, shape (lat, lon)
        Number of days in the 14-day period that experienced at least 1 mm (0.04 in) of rainfall. Filled after :func:`checkRainyDays` is called.
    totals3Day : array, shape (lat, lon)
        3-day rainfall total for the day of maximum precipitation and the two days surrounding for each point in space. Filled after :func:`check3DayTotals` is called.
    frac : array, shape (lat, lon)
        Fraction of total rainfall that fell in the 3-day period as specified in ``totals3Day``. Filled after :func:`check3DayTotals` is called.
    extreme : array, shape (lat, lon)
        True where ``diff`` is positive and ``daysOver2`` is at least 5; False if either condition is not met. Filled after :func:`getExtremePoints` is called.
    Z : array, shape (kdeGridX, kdeGridY)
        Density (height in the vertical coordinate) obtained from the KDE analysis. Filled after :func:`kde` is called.
    areas : dict
        Areas of KDE (:func:`kde`) contours drawn in square kilometers. Filled after :func:`getAreas` is called.
    """

    #global _applyMask
    _daysInMonth = {'1':['January',31], '2':['February',28], '3':['March',31],
                    '4':['April',30], '5':['May',31], '6':['June',30],
                    '7':['July',31], '8':['August',31], '9':['September',30],
                    '10':['October',31], '11':['November',30], '12':['December',31]}

    def __init__(self,month,day,year,length=14):
        #init called first
        #print('init method called')
        self.month = month
        self.day = day
        self.year = year
        self.length = length

        #FIXME: Down the road, will have 1 netCDF4 file for all 95 percentiles
        #with lengths 14, 30, 60, and 90. Take additional argument in constructor
        #to slice intercept, slope for correct length
        with Dataset('/share/data1/ty/models/quantReg.95.14.nc','r') as nc:
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

        gridLats = np.arange(24, 50.1, 0.1)
        gridLons = np.arange(232, 294.1, 0.1)
        self.kdeGridX, self.kdeGridY = np.meshgrid(gridLons, gridLats)

        self.obs = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.total = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.diff = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.daysOver2 = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.frac = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.extreme = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.Z = np.zeros_like(self.kdeGridX) * np.nan
        self.areas = {}
        self.polys = []
        self._noPoints = False

    def __repr__(self):
        return 'DateTest(month=%s, day=%s, year=%s)'%(self.month, self.day, self.year)

    @property
    def month(self):
        #property called third
        #print('month property method called')
        """Get or set the current month. Setting the month to a new value will
        set the ``DATE_BEGIN`` and ``DATE_END`` properties automatically.
        """
        return self.__month

    @month.setter
    def month(self,val):
        #setter called second; if attribute updated after initialization, only setter method called
        #print('month setter method called')
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
        elif (val > 2011):
            raise ValueError('Years after 2011 are currently not supported.')
        else:
            self.__year = val

    '''
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
    '''

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

    """
    def _mask(self, arrToMask):
        arr = xr.DataArray(arrToMask, dims=['lat','lon'],
                coords={'lat':self.lat,'lon':self.lon})

        shp = salem.read_shapefile('/share/data1/ty/NA_shapefile/North_America.shp')
        shpSlice = shp.loc[shp['NAME'].isin(['UNITED STATES'])]
        test = arr.salem.roi(shape=shpSlice)
        test = test.values
        return arrToMask
    """

    def getObs(self):
        """Retrive Livneh reanalyses from the year specified by the object.

        Creates the observations and difference attributes for the instance. Observations
        are from Livneh and the difference is the observation amounts minus the amount given
        by the quantile regression model. Furthermore, the differences are specified to be 1
        if the rainfall was greater than the extreme threshold and 0 if less than the extreme
        threshold.
        """
        #last day in which all days in the window are in the same year
        cutoffDay = datetime.datetime(self.year,12,31) - datetime.timedelta(days=self.length-1)

        if cutoffDay > self.DATE_END:
            with Dataset('/share/data1/reanalyses/Livneh/prec.'+str(self.year)+'.nc','r') as nc:
                #print('Getting observations from %s'%self.year)
                time = nc.variables['time'][:]
                timeUnits = nc.variables['time'].units
                timeCalendar = nc.variables['time'].calendar
                time = num2date(time,timeUnits,timeCalendar)
                month = np.array([d.month for d in time])
                day = np.array([d.day for d in time])
                self.obs = nc.variables['prec'][:]
                self.units = nc.variables['prec'].units

            if self.DATE_BEGIN.month == self.DATE_END.month:
                self._locs = np.where(((month==self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) & ((month==self.DATE_END.month)&(day<=self.DATE_END.day)))[0]
            else:
                self._locs = np.where(((month==self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) | ((month==self.DATE_END.month)&(day<=self.DATE_END.day)))[0]

            self.obs = self.obs[self._locs,:,:]
            self.obs = self.obs.filled(np.nan)
            self._time = time[self._locs]

        else:
            #here, the window goes into the following year so we must load two files
            with Dataset('/share/data1/reanalyses/Livneh/prec.'+str(self.DATE_BEGIN.year)+'.nc', 'r') as nc:
                time = nc.variables['time'][:]
                timeUnits = nc.variables['time'].units
                timeCalendar = nc.variables['time'].calendar
                time1 = num2date(time,timeUnits,timeCalendar)
                month = np.array([d.month for d in time1])
                day = np.array([d.day for d in time1])

                time1Locs = np.where(((month>=self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) & ((month<=12)&(day<=31)))[0]
                time1Obs = nc.variables['prec'][time1Locs,:,:]
                time1Obs = time1Obs.filled(np.nan)
                self.units = nc.variables['prec'].units

            with Dataset('/share/data1/reanalyses/Livneh/prec.'+str(self.DATE_END.year)+'.nc', 'r') as nc:
                time = nc.variables['time'][:]
                timeUnits = nc.variables['time'].units
                timeCalendar = nc.variables['time'].calendar
                time2 = num2date(time,timeUnits,timeCalendar)
                month = np.array([d.month for d in time2])
                day = np.array([d.day for d in time2])

                time2Locs = np.where(((month>=1)&(day>=1)) & ((month<=self.DATE_END.month)&(day<=self.DATE_END.day)))[0]
                time2Obs = nc.variables['prec'][time2Locs,:,:]
                time2Obs = time2Obs.filled(np.nan)

            self.obs = np.concatenate((time1Obs, time2Obs), axis=0)
            self._time = np.concatenate((time1[time1Locs],time2[time2Locs]), axis=None)

        self.total = np.sum(self.obs,axis=0)
        #self.model = self._mask(self.model)
        #self.total = self._mask(self.total)
        self.diff = self.total - self.model
        return

    def checkRainyDays(self):
        """Calculate the number of days each grid point experienced
        at least 1 mm (0.04 in) of rainfall for the given 14-day period.
        """
        #print('Checking for rainy days')
        t,y,x = self.obs.shape
        obs = self.obs.reshape(t,y*x)

        #find the number of times each column goes over 1 mm, then count the bins from 0 to the number of columns
        self.daysOver2 = np.bincount(np.where(obs >= 1)[1], minlength=obs.shape[1])
        self.daysOver2 = self.daysOver2.reshape(y,x)
        return

    def check3DayTotals(self):
        """Check if the day with the maximum precipitation and the two
        days surrounding it exceed 50% of the total rainfall received in the 14-day
        period.
        """
        #print('Checking 3-day totals around day of maximum')
        t,y,x = self.obs.shape
        obs = self.obs.reshape(t,y*x)

        nonNaN = np.where(~np.isnan(obs[0,:]))[0]
        tmpTotals3Day = np.zeros((3, obs.shape[1]))*np.nan
        for i in nonNaN:
            loc = np.argmax(obs[:,i], axis=0)
            if loc == 0:
                #use first 3 values if the max rain is on day 1
                tmpTotals3Day[:,i] = obs[:3,i]
            elif loc == (t-1):
                #use last 3 values if the max rain is on day 14; t-2 is the second to last point
                tmpTotals3Day[:,i] = obs[-3:,i]
            else:
                tmpTotals3Day[:,i] = obs[loc-1:loc+2,i]

        self.totals3Day = np.nansum(tmpTotals3Day, axis=0)
        self.frac = self.totals3Day / self.total.reshape(y*x)
        self.frac = self.frac.reshape(y,x)
        return

    def getExtremePoints(self):
        """Find which points are extreme.

        Points must have exceeded the 14-day 95th percentile, have experienced
        at least 5 days of rainfall at or exceeding 1 mm (0.04 in), and had less
        than 50% of the total rainfall fall in the day of maximum precipitation and
        the 2 surrounding days.
        """
        if np.where(~np.isnan(self.diff))[0].size == 0:
            self.getObs()
        if np.where(~np.isnan(self.daysOver2))[0].size == 0:
            self.checkRainyDays()
        if np.where(~np.isnan(self.frac))[0].size == 0:
            self.check3DayTotals()

        self.extreme = (self.diff >= 0) & (self.daysOver2 >= 5) & (self.frac <= 0.5)
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

        if np.where(~np.isnan(self.extreme))[0].size == 0:
            self.getExtremePoints()

        x,y = np.meshgrid(self.lon, self.lat)
        locs = np.ma.where(self.extreme == 1)
        if locs[0].size == 0:
            print('No extreme points')
            self._noPoints = True
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

        kwargs.setdefault('bandwidth', 0.02)
        kwargs.setdefault('metric', 'haversine')
        kwargs.setdefault('kernel', 'epanechnikov')
        kwargs.setdefault('algorithm', 'ball_tree')
        self.KDE = KernelDensity(**kwargs)
        self.KDE.fit(self._XtrainRad, sample_weight=weighted)
        self.Z = np.exp(self.KDE.score_samples(xy))
        #divide by max to normalize density field
        self.Z = self.Z / np.nanmax(self.Z)
        self.Z = self.Z.reshape(self.kdeGridX.shape)
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
        elif np.where(~np.isnan(self.Z))[0].size == 0:
            self.kde()
        return np.nanpercentile(a=self.Z, q=perc)

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
        ax.set_extent([232,294,24,50])
        self._im = plt.contour(self.kdeGridX, self.kdeGridY, self.Z, levels=levels,
                            transform=self._origProj)

        self._levels = self._im.levels
        return

    def _coordTransform(self, x, y):
        return self._targetProj.transform_points(self._origProj, x, y)

    def calcAreas(self):
        """Calculate the area of the polygons in the KDE map shown by :func:`getContours`.

        The vertices are gathered from the objected returned by matplotlib's
        `contour <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html>`_
        method and the area is calculated using Green's Theorem. Requires the :func:`getContours` method
        to have been called since it requires vertices to describe the polygon that we are
        finding the area of!

        Areas for each contour are stored in a dictionary and assigned as an instance attribute
        with units of squared kilometers.
        """
        if self._noPoints:
            return

        try:
            getattr(self, '_im')
        except AttributeError:
            if np.where(~np.isnan(self.Z))[0].size == 0:
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
                if a >= 100000.:
                    self.polys.append(Polygon([(j[0], j[1]) for j in zip(region.vertices[:,0],region.vertices[:,1])]))
            self.areas[self._levels[i]] = areas
        return

    def maskOutsideRegion(self):
        """Set all points outside extreme region to NaN.

        Additionally, arrays are created to find areal-averaged precipitation,
        maximum 1-day and 14-day precipitation, and the total precipitation over
        the extreme threshold in the extreme region.
        """
        import xarray as xr
        import salem

        if self._noPoints:
            self.Z = np.zeros_like(self.kdeGridX)
            return
        elif len(self.polys) == 0:
            print('No extreme regions')
            return
        else:
            daily = xr.DataArray(self.obs, dims=('time', 'lat', 'lon'), coords={'time':self._time, 'lat':self.lat, 'lon':self.lon})
            total = xr.DataArray(self.total, dims=('lat', 'lon'), coords={'lat':self.lat, 'lon':self.lon})
            diff = xr.DataArray(self.diff, dims=('lat', 'lon'), coords={'lat':self.lat, 'lon':self.lon})

            self.regionsDaily = np.zeros((len(self.polys), daily.shape[0], self.lat.size, self.lon.size)) * np.nan
            self.regionsTotal = np.zeros((len(self.polys), self.lat.size, self.lon.size)) * np.nan
            self.regionsDiff = np.zeros((len(self.polys), self.lat.size, self.lon.size)) * np.nan
            for i in range(len(self.polys)):
                self.regionsDaily[i,:,:,:] = daily.salem.roi(geometry=self.polys[i])
                self.regionsTotal[i,:,:] = total.salem.roi(geometry=self.polys[i])
                self.regionsDiff[i,:,:] = diff.salem.roi(geometry=self.polys[i])

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

import numpy as np
from netCDF4 import Dataset,num2date
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon
import pandas as pd
import datetime
import warnings
#import time as t
warnings.simplefilter("ignore")
import cartopy.crs as ccrs

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
    model : array, shape (lat, lon)
        95th percentile grid; calculated by doing ``intercept`` + ``slope`` x ``year`` for the input ``day``.
    kdeGridX : array, shape (261, 622)
        Longitude grid the kernel density estimation is evaluated onto. Range is [128W, 66W] every 1/10th of a degree.
    kdeGridY : array, shape (261, 622)
        Latitude grid the kernel density estimation is evaluated onto. Range is [24N, 50N] every 1/10th of a degree.
    obs : array, shape (lat, lon)
        Recorded precipitation each day from the Livneh dataset for the input 14-day period. Filled after :func:`getObs` is called.
    total : array, shape(lat, lon)
        Total precipitation from the Livneh dataset (i.e., sum at each grid point of ``obs``). Filled after :func:`getObs` is called.
    diff : array, shape (lat, lon)
        Difference between ``obs`` and ``model``. Filled after :func:`getObs` or :func:`plotExtremePoints` is called.
    daysOver2 : array, shape (lat, lon)
        Number of days in the 14-day period that experienced at least 1 mm (0.04 in) of rainfall. Filled after :func:`plotRainyDays` or :func:`plotExtremePoints` is called.
    totals3Day : array, shape (lat, lon)
        3-day rainfall total for the day of maximum precipitation and the two days surrounding for each point in space. Filled after :func:`plot3DayTotals` or :func:`plotExtremePoints` is called.
    frac : array, shape (lat, lon)
        Fraction of total rainfall that fell in the 3-day period as specified in ``totals3Day``. Filled after :func:`plot3DayTotals` or :func:`plotExtremePoints` is called.
    extreme : array, shape (lat, lon)
        True where ``diff`` is positive and ``daysOver2`` is at least 5; False if either condition is not met. Filled after :func:`plotExtremePoints` is called.
    Z : array, shape (kdeGridX, kdeGridY)
        Density (height in the vertical coordinate) obtained from the KDE analysis. Filled after :func:`kde` is called.
    areas : dict
        Areas of KDE (:func:`kde`) contours drawn in :func:`makePlot` in square kilometers. Filled after :func:`getAreas` is called.
    """


    with Dataset('/share/data1/ty/models/quantReg.95.14.nc','r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        time = nc.variables['time'][:]
        timeUnits = nc.variables['time'].units
        timeCalendar = nc.variables['time'].calendar
        time = num2date(time,timeUnits,timeCalendar)
        _month = np.array([d.month for d in time])
        _day = np.array([d.day for d in time])
        intercept = nc.variables['intercept'][:]
        slope = nc.variables['slope'][:]
        del time,timeUnits,timeCalendar,nc,d

    _daysInMonth = {'1':['January',31], '2':['February',28], '3':['March',31],
                    '4':['April',30], '5':['May',31], '6':['June',30],
                    '7':['July',31], '8':['August',31], '9':['September',30],
                    '10':['October',31], '11':['November',30], '12':['December',31]}


    def __init__(self,month,day,year):
        #init called first
        #print('init method called')
        self.month = month
        self.day = day
        self.year = year
        loc = np.where((self._month == self.month) & (self._day == self.day))[0][0]
        self.model = self.intercept[loc,:,:] + self.slope[loc,:,:]*self.year

        gridLats = np.arange(24, 50.1, 0.1)
        gridLons = np.arange(232, 294.1, 0.1)
        self.kdeGridX, self.kdeGridY = np.meshgrid(gridLons, gridLats)

        self.obs = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.total = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.diff = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.daysOver2 = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.frac = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.extreme = np.zeros((self.lat.size, self.lon.size)) * np.nan
        self.Z = np.zeros((self.lat[::3].size, self.lon[::3].size)) * np.nan
        self.areas = {}
        self.polys = []

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

    def getObs(self):
        """Method to retrive Livneh reanalyses from the year specified by the object.

        Creates the observations and difference attributes for the instance. Observations
        are from Livneh and the difference is the observation amounts minus the amount given
        by the quantile regression model. Furthermore, the differences are specified to be 1
        if the rainfall was greater than the extreme threshold and 0 if less than the extreme
        threshold.
        """
        with Dataset('/share/data1/reanalyses/Livneh/prec.'+str(self.year)+'.nc','r') as nc:
            #print('Getting observations from %s'%self.year)
            time = nc.variables['time'][:]
            timeUnits = nc.variables['time'].units
            timeCalendar = nc.variables['time'].calendar
            time = num2date(time,timeUnits,timeCalendar)
            month = np.array([d.month for d in time])
            day = np.array([d.day for d in time])
            self.obs = nc.variables['prec'][:]
            if self.DATE_BEGIN.month == self.DATE_END.month:
                locs = np.where(((month==self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) & ((month==self.DATE_END.month)&(day<=self.DATE_END.day)))[0]
            else:
                locs = np.where(((month==self.DATE_BEGIN.month)&(day>=self.DATE_BEGIN.day)) | ((month==self.DATE_END.month)&(day<=self.DATE_END.day)))[0]

            self.obs = self.obs[locs,:,:]
            self.obs = self.obs.filled(np.nan)
            self.total = np.sum(self.obs,axis=0)
            self.diff = self.total - self.model
        return

    def checkRainyDays(self):
        """Helper method to calculate the number of days each grid point experienced
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
        """Helper method to check if the day with the maximum precipitation and the two
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
        """Helper method to find which points are extreme.

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
        """Method to calculate the kernel density estimate for a given period.

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
        XtrainRad = Xtrain * np.pi / 180.
        xy = np.vstack((self.kdeGridY.ravel(), self.kdeGridX.ravel())).T
        xy *= np.pi / 180.

        kwargs.setdefault('bandwidth', 0.02)
        kwargs.setdefault('metric', 'haversine')
        kwargs.setdefault('kernel', 'epanechnikov')
        kwargs.setdefault('algorithm', 'ball_tree')
        self.kde = KernelDensity(**kwargs)
        self.kde.fit(XtrainRad, sample_weight=weighted)
        self.Z = np.exp(self.kde.score_samples(xy))
        self.Z = self.Z.reshape(self.kdeGridX.shape)
        return

    def calcKDEPercentile(self, perc=95):
        """Method to return a KDE density for a given percentile.

        Parameters
        ----------
        perc : float in range of [0,100]
            Percentile to find in the KDE distribution; must be between 0 and 100, inclusive (defaults to 95)
        """
        if np.where(~np.isnan(self.Z))[0].size == 0:
            self.kde()
        return np.nanpercentile(a=self.Z, q=perc)

    def plotKDEDistribution(self):
        """Method to plot a histogram of the KDE densities.
        """
        if np.where(~np.isnan(self.Z))[0].size == 0:
            self.kde()

        fig = plt.figure(figsize=(8,6))
        plt.hist(self.Z.ravel(), bins=np.arange(0, np.round(np.max(self.Z))+1, 1), density=True,
                color='deepskyblue', ec='k')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('KDE Density', fontsize=13)
        plt.ylabel('Relative Frequency', fontsize=13)
        plt.title('Distribution of KDE Densities', fontsize=15)
        plt.show(block=False)
        return

    def getContours(self):
        ax = plt.axes(projection=ccrs.AlbersEqualArea(central_latitude=35, central_longitude=250))
        ax.set_extent([232,294,24,50])
        self._im = plt.contour(self.kdeGridX, self.kdeGridY, self.Z, levels=[self.calcKDEPercentile()],
                            transform=ccrs.AlbersEqualArea)
        self._levels = self._im.levels
        return

    def calcAreas(self):
        """Method to calculate the area of the polygons in the KDE map shown by makePlot.
        The vertices are gathered from the objected returned by matplotlib's
        `contour <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html>`_
        method and the area is calculated using Green's Theorem. Requires the :func:`makePlot` method
        to have been called since it requires vertices to describe the polygon that we are
        finding the area of!

        Areas for each contour are stored in a dictionary and assigned as an instance attribute
        with units of squared kilometers.
        """
        try:
            getattr(self, '_im')
        except AttributeError:
            if np.where(~np.isnan(self.Z))[0].size == 0:
                self.kde()
            self.getContours()

        numContours = len(self._im.collections)
        self.areas = {}
        #self.polys = []
        for i in range(numContours):
            areas = []
            for region in self._im.collections[i].get_paths():
                x = region.vertices[:,0]
                y = region.vertices[:,1]
                a = 0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
                a = np.abs(a) / (1000.**2)
                areas.append(a)
                #if a >= 100000.:
                    #lons, lats = self._m(x, y, inverse=True)
                    #self.polys.append(Polygon([(j[0], j[1]) for j in zip(lons,lats)]))
            self.areas[self._levels[i]] = areas
        return

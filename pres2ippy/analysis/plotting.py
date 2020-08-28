import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.basemap import Basemap
import os
import colormaps
import datetime

wdir = os.getcwd()

class InputError(Exception):
    pass

def plotSlopes(obj, endpoints, ticks):
    """Plot the slope of the quantile regression model for the
    input 14-day period.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    ends : list or tuple
        Endpoints on the colorbar
    ticks : int
        Spacing between ticks on the colorbar
    """
    x,y = np.meshgrid(obj.lon, obj.lat)
    bounds, my_cmap, norm = colormaps.slopes(endpoints, ticks)

    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.contourf(x,y,obj.slope, latlon=True, levels=bounds, cmap=my_cmap, norm=norm, extend='both')
    tickLabels = [endpoints[0] + 0.5*i for i in range(im.levels.size)]
    cbar = m.colorbar(im, 'bottom', ticks=tickLabels)
    cbar.set_label('mm/year')
    plt.title('%d/%d - %d/%d 95th Percentile Slope'
        %(obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_END.month,obj.DATE_END.day))
    plt.tight_layout()
    plt.show(block=False)
    return
'''
def plotRainyDays(obj, **kwargs):
    """Plot the number of days that experienced at least 1 mm (0.04 in) of precipitation
    for the given 14-day period.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    **kwargs
        Additional keyword arguments accepted by matplotlib's `contourf <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`_ Function.
    """
    if np.where(~np.isnan(obj.daysOver2))[0].size == 0:
        obj.checkRainyDays()

    x,y = np.meshgrid(obj.lon,obj.lat)
    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.contourf(x, y, obj.daysOver2, latlon=True, **kwargs)
    m.colorbar(im, 'bottom')
    plt.tight_layout()
    plt.show(block=False)
    return

def plotShortTermTotals(obj, **kwargs):
    """Plot the fraction of precipitation that fell on the day of maximum precipitation
    and the two days surrounding.

    Uses the first 3 days if the day of the maximum was day 1 of the event; uses
    the last 3 days if the day of the maximum was day 14 of the event.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    **kwargs
        Additional keyword arguments accepted by matplotlib's `contourf <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`_ Function.
    """
    if np.where(~np.isnan(obj.frac))[0].size == 0:
        obj.check3DayTotals()

    kwargs.setdefault('levels', np.arange(0,1.1,0.1))

    x,y = np.meshgrid(obj.lon,obj.lat)
    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.contourf(x, y, np.round(obj.frac,3), latlon=True, **kwargs)
    m.colorbar(im, 'bottom')
    plt.tight_layout()
    plt.show(block=False)
    return
'''
def plotDuration(obj, **kwargs):
    """Plot the number of days in the period that exceeded the mean daily precipitation.
    Mean daily precipitation calculated between 1915-2011 from Livneh.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    **kwargs
        Additional keyword arguments accepted by matplotlib's `contourf <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`_ Function.
    """
    if np.where(~np.isnan(obj.duration))[0].size == 0:
        obj.checkDuration()

    kwargs.setdefault('levels', np.arange(0,15,1))
    kwargs.setdefault('cmap', 'GnBu')

    x,y = np.meshgrid(obj.lon,obj.lat)
    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.contourf(x, y, obj.duration, latlon=True, **kwargs)
    #FIXME: check obj to pull out number of duration days
    m.contour(x, y, obj.duration, latlon=True, levels=[7], colors='k')
    m.colorbar(im, 'bottom')
    plt.title('%d/%d - %d/%d Number of Days Above Daily Mean Precipitation'
        %(obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_END.month,obj.DATE_END.day))
    plt.tight_layout()
    plt.show(block=False)

def plotExtremePoints(obj):
    """Plot the points that are labeled as extreme.

    Extreme points are colored green while non-extreme points are colored white.
    A point is labeled as extreme if its 14-day total precipitation exceeded the 95th
    percentile, it experienced at least 5 days of precipitation of at least 1 mm (0.04 in),
    and it did not have more than 50% of the total precipitation fall on the day
    of maximum precipitation and the surrounding 2 days.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    """
    if np.where(~np.isnan(obj.extreme))[0].size == 0:
        obj.getExtremePoints()

    x,y = np.meshgrid(obj.lon,obj.lat)
    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    cmap = colors.ListedColormap(['white','green'])
    im = m.pcolormesh(x, y, obj.extreme, cmap=cmap, latlon=True)
    plt.title('%s %d/%d - %d/%d Extreme Points'
        %(obj.datasetBegin,obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_END.month,obj.DATE_END.day))
    plt.tight_layout()
    plt.show(block=False)
    return
"""
def area(im):
    numContours = len(im.collections)
    areasDict = {}
    #self.polys = []
    for i in range(numContours):
        areas = []
        for region in im.collections[i].get_paths():
            x=region.vertices[:,0]
            y=region.vertices[:,1]
            a = 0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
            a = np.abs(a) / (1000.**2)
            areas.append(a)
            #if a >= 100000.:
                #lons, lats = self._m(x, y, inverse=True)
                #self.polys.append(Polygon([(j[0], j[1]) for j in zip(lons,lats)]))
        areasDict[im.levels[i]] = areas
    print(areasDict)
    return
"""

def plotKDEDensity(obj, **kwargs):
    """Plot the kernel density estimation normalized density.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    """

    kwargs.setdefault('levels', np.arange(0,1.1,0.1))
    kwargs.setdefault('cmap', 'Reds')

    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    ticks =
    im = m.contourf(obj.kdeGridX, obj.kdeGridY, obj.density, latlon=True, **kwargs)
    cbar = m.colorbar(im,'bottom')
    cbar.set_label('Normalized Density')
    plt.title('%s %d/%d - %d/%d KDE with %s Kernel and %s Bandwidth'
        %(obj.datasetBegin,obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_END.month,obj.DATE_END.day,obj.KDE.kernel.capitalize(), obj.KDE.bandwidth))
    plt.tight_layout()
    plt.show(block=False)
    return

def makePlot(obj, filled=True, cmapRain='greg', save=False, **kwargs):
    """Make 3- or 4-panel plot based on instance attributes.

    Top left panel will always be the precipitation given by the Livneh dataset.
    Top right panel will always be the thresholds for extreme given by the quantile
    regression model. Bottom left panel will always be the difference, with green shading
    where there was extreme precipitation and white otherwise. Bottom right panel is an optional panel,
    being the KDE smoothed map.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    filled : boolean
        If True (default), plot the KDE map as filled contours. Otherwise, do not fill.
    cmapRain : str
        Type of colormap to use for the top 2 panels. Current options are a colormap made by
        Ty Dickinson ('ty'), Greg Jennrich ('greg'), a copy of the colormap used on the PRISM website ('prism'),
        or a similar copy of a radar reflectivity type colormap ('custom').
    save : boolean
        If True, save plots to a temporary path. Defaults to False.
    **kwargs
        Additional keyword arguments accepted by matplotlib's `contour <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html>`_
        Function and matplotlib's `contourf <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`_ Function.
    """
    kwargs.setdefault('levels', np.arange(0,1.1,0.1))

    cmapOptions = ['ty','greg','prism','custom']
    if cmapRain.lower() not in cmapOptions:
        raise InputError('%s is not a current option. Options are %s'%(cmapRain,cmapOptions))

    x,y = np.meshgrid(obj.lon,obj.lat)
    boundsPrecip, cmapPrecip, normPrecip = colormaps.precipitation(kind=cmapRain.lower())

    fig = plt.figure(figsize=(13,10))
    for plot_num, contour in enumerate([obj.total,obj.model,obj.extreme]):
        ax = fig.add_subplot(int('22'+str(plot_num+1)))
        m = Basemap(projection='aea',resolution='l',
            llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
            lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        if (plot_num == 0):
            im = m.contourf(x,y,contour,levels=boundsPrecip,cmap=cmapPrecip,norm=normPrecip,extend='max',latlon=True)
            ax.set_title('(a) %d/%d/%d - %d/%d/%d Observed Precipitation'
                %(obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_BEGIN.year,obj.DATE_END.month,obj.DATE_END.day,obj.DATE_END.year))
            cbar = m.colorbar(im,'bottom')
            cbar.set_label('mm')
        elif (plot_num == 1):
            im = m.contourf(x,y,contour,levels=boundsPrecip,cmap=cmapPrecip,norm=normPrecip,extend='max',latlon=True)
            ax.set_title('(b) %d/%d/%d - %d/%d/%d 95th Percentile'
                %(obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_BEGIN.year,obj.DATE_END.month,obj.DATE_END.day,obj.DATE_END.year))
            cbar = m.colorbar(im,'bottom')
            cbar.set_label('mm')
        else:
            im = m.contourf(x,y,contour,levels=[0,0.5,1],colors=['white','green'],latlon=True)
            ax.set_title('(c) Extreme Points')

    if np.where(~np.isnan(obj.density))[0].size != 0:
        ax = fig.add_subplot(224)
        m = Basemap(projection='aea',resolution='l',
            llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
            lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        if filled:
            im = m.contourf(obj.kdeGridX, obj.kdeGridY, obj.density, latlon=True, **kwargs)
            cbar = m.colorbar(im,'bottom')
            cbar.set_label('Normalized Density')
        else:
            im = m.contour(obj.kdeGridX, obj.kdeGridY, obj.density, latlon=True, **kwargs)
            #plt.clabel(obj._im, fmt='%1.0f', fontsize='small')

        ax.set_title('(d) KDE with %s Kernel and %s Bandwidth'%(obj.KDE.kernel.capitalize(), obj.KDE.bandwidth))
    else:
        pass

    plt.tight_layout()
    plt.show(block=False)
    if save:
        fig.savefig('%s/tmp/%d.%d.%d.png'%(wdir,obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_BEGIN.year))
    #area(im)
    return

def plotKDEDistribution(obj):
    """Plot a histogram of the KDE densities.
    """
    if np.where(~np.isnan(obj.density))[0].size == 0:
        obj.kde()

    fig = plt.figure(figsize=(8,6))
    plt.hist(obj.density.ravel(), bins=np.arange(0, 1.01, 0.05), density=True,
            color='deepskyblue', ec='k')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('KDE Density', fontsize=13)
    plt.ylabel('Relative Frequency', fontsize=13)
    plt.title('Distribution of KDE Densities', fontsize=15)
    plt.show(block=False)
    return

def plotObsAndRegion(obj, level, cmap='greg'):
    """Plot observed precipitation with bounding region laid on top.
    """

    cmapOptions = ['ty','greg','prism','custom']
    if cmap.lower() not in cmapOptions:
        raise InputError('%s is not a current option. Options are %s'%(cmap,cmapOptions))

    x,y = np.meshgrid(obj.lon,obj.lat)
    boundsPrecip, cmapPrecip, normPrecip = colormaps.precipitation(kind=cmap.lower())

    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.contourf(x,y,obj.total,levels=boundsPrecip,cmap=cmapPrecip,norm=normPrecip,extend='max',latlon=True)
    cbar = m.colorbar(im, 'bottom')
    cbar.set_label('mm')
    m.contour(obj.kdeGridX, obj.kdeGridY, obj.density, levels=[level], colors='r', latlon=True)
    plt.title('%d/%d/%d - %d/%d/%d Observed Precipitation and Extreme Designations'
        %(obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_BEGIN.year,obj.DATE_END.month,obj.DATE_END.day,obj.DATE_END.year))
    plt.tight_layout()
    plt.show(block=False)
    return

def plotObsEvolution(obj,cmap='greg'):
    x,y = np.meshgrid(obj.lon, obj.lat)
    boundsPrecip, cmapPrecip, normPrecip = colormaps.precipitation(kind=cmap.lower(), bounds=np.linspace(0,160,17))

    fig = plt.figure(figsize=(15,7))
    for i in range(obj.length):
        ax = fig.add_subplot(3,5,i+1)
        date = obj.DATE_BEGIN + datetime.timedelta(days=i)
        m = Basemap(projection='aea',resolution='l',
            llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
            lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        im = m.contourf(x,y,obj.obs[i,:,:],levels=boundsPrecip,cmap=cmapPrecip,norm=normPrecip,extend='max',latlon=True)
        plt.title('%d/%d/%d'%(date.month, date.day, date.year))

    ax = fig.add_subplot(3,5,15)
    ax.set_axis_off()
    cbar = fig.colorbar(im, orientation='horizontal', ax=ax)
    cbar.set_label('mm')
    plt.tight_layout()
    plt.show(block=False)
    return

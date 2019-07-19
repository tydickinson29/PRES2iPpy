import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap

def plotSlopes(obj):
    """Function to plot the slope of the quantile regression model for the
    input 14-day period.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    """
    x,y = np.meshgrid(obj.lon, obj.lat)
    cmap = cm.get_cmap('BrBG')
    my_colors = [
        cmap(1./20), #-1 to -0.9
        cmap(2./20), #-0.9 to -0.8
        cmap(3./20), #-0.8 to -0.7
        cmap(4./20), #-0.7 to -0.6
        cmap(5./20), #-0.6 to -0.5
        cmap(6./20), #-0.5 to -0.4
        cmap(7./20), #-0.4 to -0.3
        cmap(8./20), #-0.3 to -0.2
        cmap(9./20), #-0.2 to -0.1
        'white', #-0.1 to 0
        'white', #0 to 0.1
        cmap(11./20), #0.1 to 0.2
        cmap(12./20), #0.2 to 0.3
        cmap(13./20), #0.3 to 0.4
        cmap(14./20), #0.4 to 0.5
        cmap(15./20), #0.5 to 0.6
        cmap(16./20), #0.6 to 0.7
        cmap(17./20), #0.7 to 0.8
        cmap(18./20), #0.8 to 0.9
        cmap(19./20) #0.9 to 1.0
    ]
    bounds = np.arange(-1,1.1,0.2)
    my_cmap = colors.ListedColormap(my_colors)
    my_cmap.set_under(cmap(0.0))
    my_cmap.set_over(cmap(1.0))
    norm = colors.BoundaryNorm(bounds, my_cmap.N)

    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    loc = np.where((obj._month == obj.month) & (obj._day == obj.day))[0][0]
    im = m.contourf(x,y,obj.slope[loc,:,:], latlon=True, levels=bounds, cmap=my_cmap, norm=norm, extend='both')
    cbar = m.colorbar(im, 'bottom')
    cbar.set_label('mm/year')
    plt.tight_layout()
    plt.show(block=False)
    return

def plotRainyDays(obj, **kwargs):
    """Function to plot the number of days that experienced at least 1 mm (0.04 in)
    of rainfall for the given 14-day period.

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

def plot3DayTotals(obj, **kwargs):
    """Function to plot the fraction of rainfall that fell on the day of maximum
    precipitation and the two days surrounding. Uses the first 3 days if the day
    of the maximum was day 1 of the event; uses the last 3 days if the day of the
    maximum was day 14 of the event.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    **kwargs
        Additional keyword arguments accepted by matplotlib's `contourf <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`_ Function.
    """
    if np.where(~np.isnan(obj.frac))[0].size == 0:
        obj.check3DayTotals()

    x,y = np.meshgrid(obj.lon,obj.lat)
    fig = plt.figure(figsize=(8,6))
    m = Basemap(projection='aea',resolution='l',
        llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
        lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.contourf(x, y, obj.frac, latlon=True, **kwargs)
    m.colorbar(im, 'bottom')
    plt.tight_layout()
    plt.show(block=False)
    return

def plotExtremePoints(obj):
    """Function to plot the points that are labeled as extreme. Extreme points are colored
    green while non-extreme points are colored white. A point is labeled as extreme if
    its 14-day total rainfall exceeded the 95th percentile, it experienced at least 5 days
    of rainfall of at least 1 mm (0.04 in), and it did not have more than 50% of the total
    precipitation fall on the day of maximum rainfall and the surrounding 2 days.

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
    im = m.contourf(x, y, obj.extreme, latlon=True, levels=[0,0.5,1], colors=['white','green'])
    m.colorbar(im, 'bottom')
    plt.tight_layout()
    plt.show(block=False)
    return

def makePlot(obj, filled=True, **kwargs):
    """Function to make 3- or 4-panel plot based on instance attributes.

    Top left panel will always be the rainfall given by the Livneh dataset.
    Top right panel will always be the thresholds for extreme given by the quantile
    regression model. Bottom left panel will always be the difference, with green shading
    where there was extreme rainfall and white otherwise. Bottom right panel is an optional panel,
    being the KDE smoothed map.

    Parameters
    ----------
    obj : object
        :class:`rainpy` object for the desired 14-day period.
    filled : boolean
        If True (default), plot the KDE map as filled contours. Otherwise, do not fill.
    **kwargs
        Additional keyword arguments accepted by matplotlib's `contour <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html>`_
        Function and matplotlib's `contourf <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html>`_ Function.
    """

    x,y = np.meshgrid(obj.lon,obj.lat)
    boundsPrecip = np.linspace(0,600,17)
    colorsPrecip = ['w','cornflowerblue','b','teal','g','yellow','gold','orange',
            'darkorange','r','crimson','darkred','k','grey','darkgrey','lightgray']
    cmapPrecip = colors.ListedColormap(colorsPrecip)
    cmapPrecip.set_over('gainsboro')
    normPrecip = colors.BoundaryNorm(boundsPrecip, cmapPrecip.N)

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
            ax.set_title('(a) %s/%s/%s - %s/%s/%s Observed Precipitation'
                %(obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_BEGIN.year,obj.DATE_END.month,obj.DATE_END.day,obj.DATE_END.year))
            cbar = m.colorbar(im,'bottom')
            cbar.set_label('mm')
        elif (plot_num == 1):
            im = m.contourf(x,y,contour,levels=boundsPrecip,cmap=cmapPrecip,norm=normPrecip,extend='max',latlon=True)
            ax.set_title('(b) %s/%s/%s - %s/%s/%s 95th Percentile'
                %(obj.DATE_BEGIN.month,obj.DATE_BEGIN.day,obj.DATE_BEGIN.year,obj.DATE_END.month,obj.DATE_END.day,obj.DATE_END.year))
            cbar = m.colorbar(im,'bottom')
            cbar.set_label('mm')
        else:
            im = m.contourf(x,y,contour,levels=[0,0.5,1],colors=['white','green'],latlon=True)
            ax.set_title('(c) Extreme Points')

    if np.where(~np.isnan(obj.Z))[0].size != 0:
        ax = fig.add_subplot(224)
        #making the Basemap object a private attribute to be used in _makeDataframe()
        m = Basemap(projection='aea',resolution='l',
            llcrnrlat=22.5,llcrnrlon=-120.,urcrnrlat=49.,urcrnrlon=-64,
            lat_1=29.5, lat_2=45.5, lat_0=37.5, lon_0=-96.)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        if filled:
            im = m.contourf(obj.kdeGridX, obj.kdeGridY, obj.Z, latlon=True, **kwargs)
            cbar = m.colorbar(im,'bottom')
            cbar.set_label('Density')
        else:
            im = m.contour(obj.kdeGridX, obj.kdeGridY, obj.Z, latlon=True, **kwargs)
            #plt.clabel(obj._im, fmt='%1.0f', fontsize='small')

        ax.set_title('(d) KDE with %s Kernel and %s Bandwidth'%(obj.kde.kernel.capitalize(), obj.kde.bandwidth))
    else:
        pass

    plt.tight_layout()
    plt.show(block=False)
    return

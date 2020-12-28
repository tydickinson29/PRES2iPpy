import numpy as np

def interp(datain, xin, yin, xout, yout):
    '''
    ALL CREDIT FOR THIS METHOD GOES TO THE DEVELOPERS OF THE BASEMAP
    PACKAGE. I use their code here since I am unable to import it in the
    conda environment used in generating the database. Later updates will change
    this method as to not copy/paste code.
    '''
    # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]
    if max(delx)-min(delx) < 1.e-4 and max(dely)-min(dely) < 1.e-4:
        # regular input grid.
        xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
        ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])
    else:
        # irregular (but still rectilinear) input grid.
        xoutflat = xout.flatten(); youtflat = yout.flatten()
        ix = (np.searchsorted(xin,xoutflat)-1).tolist()
        iy = (np.searchsorted(yin,youtflat)-1).tolist()
        xoutflat = xoutflat.tolist(); xin = xin.tolist()
        youtflat = youtflat.tolist(); yin = yin.tolist()
        xcoords = []; ycoords = []
        for n,i in enumerate(ix):
            if i < 0:
                xcoords.append(-1) # outside of range on xin (lower end)
            elif i >= len(xin)-1:
                xcoords.append(len(xin)) # outside range on upper end.
            else:
                xcoords.append(float(i)+(xoutflat[n]-xin[i])/(xin[i+1]-xin[i]))
        for m,j in enumerate(iy):
            if j < 0:
                ycoords.append(-1) # outside of range of yin (on lower end)
            elif j >= len(yin)-1:
                ycoords.append(len(yin)) # outside range on upper end
            else:
                ycoords.append(float(j)+(youtflat[m]-yin[j])/(yin[j+1]-yin[j]))
        xcoords = np.reshape(xcoords,xout.shape)
        ycoords = np.reshape(ycoords,yout.shape)

    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)
    xi = xcoords.astype(np.int32)
    yi = ycoords.astype(np.int32)
    xip1 = xi+1
    yip1 = yi+1
    xip1 = np.clip(xip1,0,len(xin)-1)
    yip1 = np.clip(yip1,0,len(yin)-1)
    delx = xcoords-xi.astype(np.float32)
    dely = ycoords-yi.astype(np.float32)
    dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
            delx*dely*datain[yip1,xip1] + \
            (1.-delx)*dely*datain[yip1,xi] + \
            delx*(1.-dely)*datain[yi,xip1]
    return dataout

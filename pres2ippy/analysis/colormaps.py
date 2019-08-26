from matplotlib import cm, colors
import numpy as np

def rainfall(kind):
    if kind == 'greg':
        bounds = np.linspace(0,600,17)
        my_colors = ['w','cornflowerblue','b','teal','g','yellow','gold','orange',
                'darkorange','r','crimson','darkred','k','grey','darkgrey','lightgray']
        my_cmap = colors.ListedColormap(my_colors)
        my_cmap.set_over('gainsboro')
        norm = colors.BoundaryNorm(bounds, cmap.N)

    elif kind == 'prism':
        bounds = [0,0.01,0.1,0.2,0.4,0.6,0.8,1.2,1.6,2,2.4,2.8,3.2,4,5,6,8,12,16,20]
        bounds = [i*25.4 for i in bounds]
        my_colors = ['#ffffff','#810000','#ae3400','#e46600','#ff9600','#ffc900',
                        '#fffb00','#c0ff00','#68fa00','#12ff2c','#2cff8e','#15ffe9',
                        '#33b6ff','#335eff','#2100ff','#8d00ff','#de00ff','#ff56ff',
                        '#ffaeff']
        my_cmap = colors.ListedColormap(my_colors)
        my_cmap.set_over('#ffd0ff')
        norm = colors.BoundaryNorm(bounds, my_cmap.N)

    elif kind == 'custom':
        my_colors = [
            (104,120,141),
            (52,80,133),
            (70,115,151),
            (88,152,171),

            (41,183,77),
            (22,143,40),
            (29,116,2),
            (156,192,0),

            (252,253,0),
            (234,229,0),
            (212,199,0),
            (198,180,0),

            (246,145,0),
            (228,130,0),
            (197,104,2),
            (179,88,4),

            (245,30,4),
            (199,32,11),
            (161,33,19),
            (131,35,26),

            (202,150,178),
            (200,106,156),
            (197,56,131),
            (195,22,115),

            (150,28,220),
            (121,22,194),
            (85,12,163),
            (63,7,144)
        ]
        my_colors = [tuple([i / 255. for i in j]) for j in my_colors]
        over = (133,253,255)
        over = tuple([i / 255. for i in over])
        my_colors.insert(0, 'white')
        bounds = np.arange(0,726,25)
        my_cmap = colors.ListedColormap(my_colors)
        my_cmap.set_over(over)
        norm = colors.BoundaryNorm(bounds, my_cmap.N)
        
    return bounds, my_cmap, norm

def slopes():
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
    bounds = np.arange(-1,1.1,0.1)
    my_cmap = colors.ListedColormap(my_colors)
    my_cmap.set_under(cmap(0.0))
    my_cmap.set_over(cmap(1.0))
    norm = colors.BoundaryNorm(bounds, my_cmap.N)
    return bounds, my_cmap, norm

# Defining Extreme

This directory is dedicated to programs written to calculate and store our two criteria in defining extreme.

## Criteria
1. A point's total precipitation must meet or exceed its length-day (e.g., 14-day, 30-day, etc) 99th percentile for a given calendar window.
2. Half of the days in the window must meet or exceed the long-term mean daily precpitation for the window.

## Files
calcWindowStats.py --> Saves individual .npy files for windows beginning on each calendar day. Takes in 2 arguments: length and mode. Length should be an int specifying the total number of days in the window. Mode should be a string and has two options. If the mode is given as "totals", precipitation is summed for each year and saved with dimensionality (time, lat, lon). If the mode is "means", the mean of all days in the window is calculated and saved with dimensionality (lat, lon).

calcPercentiles.py --> Calculates an input percentile, smoothes using a Fourier transform, and saves the percentiles in netCDF4 format. Takes in 3 arguments: length, percentile, and components. Length is same as in calcWindowStats. Percentile should be an int (but also can be a float) telling the program what percentile of precipitation to calculate. Components should be an int determining the number of Fourier harmonics to keep. For example, `--components=3` will use wavenumbers 0, 1, 2, and 3 to smooth the raw signal. The percentiles are finally saved into a netCDF4 file with a filename with the dataset name, the percentile (forced to an int), and the length included.

makeDurationFile.py --> Loads the saved mean files from calcWindowStats, concatenates them together, and saves the data in a netCDF4 file. Takes in 1 argument: length. The filename includes the dataset name and the length.


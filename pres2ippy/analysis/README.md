# Directory Info

This directory houses files used to make the databases.

## Description of Files
rainpy.py --> The main file used in building the database. The program contains a class, DateTest, which is constructed by inputting a start month, day, and year, as well as window length, percentile, and dataset. Methods for the class include loading the appropriate data, regridding if needed, getting extreme grid points using our two criteria, performing kernel density estimation, and making arrays with grid points outside an extreme polygon being masked.

buildKDEDistribution.py --> Uses rainpy.py to build a distribution of KDE densities for all `length`-day windows for an input month across all years in `dataset` period of record. Saves distribution as one .npy file. MPI is implemented so the script can be run in parallel. We use the 12 files (one for each month) to build the entire KDE distribution to find the 99th percentile to use as the bounding contour for extreme polygons.

buildAreaDistribution.py --> Builds on buildKDEDistribution.py by using the contour to develop polygons and the area of each polygon is saved. We use the 12 files to find the ~95th percentile to use as the area threshold for the database.

generateDatabase.py --> Combines utilization of both KDE density contour and area threshold to build the database. Any polygon with an area > the threshold determined by the previous step is saved, along with various information such as area-averaged precipitation, total over extreme, and geospatial information. After all dates are tested, the info is put into a Pandas DataFrame and saved in .csv format.

postprocess.py --> Postprocessing algorithm for the database. Groups polygons into similar groups if they have overlapping windows and spatial correlations of >= 0.5. See Dickinson et al. (2021) for more details. Saves the final database in both .csv and .shp formats.

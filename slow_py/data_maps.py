import csep
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gdp
from catalogs import *
from slab import Slab, ALL_SLABS

regions = ['mexico'
           'japan']
slabs_per_region = {
    'mexico': [Slab('cam',  path='../data/Slab2/xyz')]
}
slabpoly_per_region = {
    'mexico': '../data/poly4seismicity/FinalPolygons_150km/mexico.shp'
}
catalogs = {
    'mexico': [get_mexico_raw_catalog(), get_mexico_catalog(depth=150)]}

filename = {'mexico': 'mexico.png'}
extents = {'mexico': [-108.984, -92.434, 11, 23.021]}
fig_sizes = {'mexico': (10, 8)}


trench = pandas.read_csv('../data/trench_data.csv', index_col=0)

for region in regions:

    ax = csep.utils.plots.plot_basemap('ESRI_terrain', extents[region], figsize=fig_sizes[region])
    # ax = csep.utils.plots.plot_basemap('stock_img', extents[region], figsize=fig_sizes[region])

    for lon1,lon2,lat1,lat2 in zip(trench.Lon1, trench.Lon2, trench.Lat1, trench.Lat2):
        if lon1 > 180:
            lon1 -= 360
        if lon2 > 180:
            lon2 -= 360
        ax.plot([lon1, lon2], [lat1, lat2], color='black', linewidth=0.5)

    ### Slab Depths
    slabs = slabs_per_region[region]
    for i in slabs:

        midpoints = np.column_stack((i.longitude[i.indices], i.latitude[i.indices]))
        csep_region = csep.core.regions.CartesianGrid2D.from_origins(midpoints)
        data = i.depth[i.indices] / 1000.
        ax = csep.utils.plots.plot_gridded_dataset(csep_region.get_cartesian(data), csep_region, ax=ax,
                                                   colormap='magma', alpha=0.5, clim=[None, 200], clabel='Depth [km]')

    ### Slab poly
    gdf = gpd.read_file(slabpoly_per_region[region])
    polygon = gdf.geometry.iloc[0]
    coords = np.array((polygon.exterior.coords))
    ax.plot(coords[:, 0], coords[:, 1], '-', color='gold')

    ### Fast Catalog

    catalogs[region][0].plot(ax=ax, markercolor='steelblue', markeredgecolor='darkblue', size=1, power=2)


    ### Slowslip
    get_slow_slip_catalog().plot(ax=ax, markercolor='red',
                                size=1, power=2,
                                 min_val=catalogs[region][0].get_magnitudes().min(),
                                 max_val=catalogs[region][0].get_magnitudes().max(),
                                 mag_ticks=np.arange(np.floor(catalogs[region][0].get_magnitudes().min()),
                                                     np.ceil(catalogs[region][0].get_magnitudes().max()))
                                 )


    plt.savefig(filename[region])

    # lon_min = np.min(i.longitudes)
        # lon_max = np.max(i.longitudes)
        # lat_min = np.min(i.latitudes)
        # lat_max = np.max(i.latitudes)
        # dx = np.diff(np.unique(i.longitudes))[0]
        # dy = np.diff(np.unique(i.latitudes))[0]
        #
        # nx = int(np.round((lon_max - lon_min) / dx, 0)) + 1
        # ny = int(np.round((lat_max - lat_min) / dy, 0)) + 1




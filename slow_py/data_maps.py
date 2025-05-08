import csep
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gdp
from catalogs import *
from slab import Slab, ALL_SLABS
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


regions = [
    'mexico',
    'japan',
    'chile',
    'alaska',
    'cascadia'
]
slabs_per_region = {
    'mexico': [Slab('cam',  path='../data/Slab2/xyz')],
    'japan': [Slab('ryu',  path='../data/Slab2/xyz'),
              Slab('izu',  path='../data/Slab2/xyz'),
              Slab('kur',  path='../data/Slab2/xyz')],
    'chile': [Slab('sam', path='../data/Slab2/xyz')],
    'alaska': [Slab('alu', path='../data/Slab2/xyz')],
    'cascadia': [Slab('cas', path='../data/Slab2/xyz')]
}
slabpoly_per_region = {
    'mexico': '../data/poly4seismicity/FinalPolygons_150km/mexico.shp',
    'japan': '../data/poly4seismicity/FinalPolygons_150km/japan.shp',
    'chile': '../data/poly4seismicity/FinalPolygons_150km/chile.shp',
    'alaska': '../data/poly4seismicity/FinalPolygons_150km/alu.shp',
    'cascadia': '../data/poly4seismicity/FinalPolygons_150km/cas.shp'

}
catalogs = {
    'mexico': [get_mexico_raw_catalog(), get_mexico_catalog(depth=150)],
    'japan': [ get_japan_raw_catalog(), get_japan_catalog(depth=150)],
    'chile': [get_chile_raw_catalog(), get_chile_catalog(depth=150)],
    'alaska': [get_alaska_catalog()],
    'cascadia': [get_cascadia_raw_catalog()]

}

filename = {
    'mexico': '../images/data_mexico_map.png',
            'japan': '../images/data_japan_map.png',
    'chile': '../images/data_chile_map.png',
     'alaska': '../images/data_alaska_map.png',
   'cascadia': '../images/data_cascadia_map.png'

}
extents = {
    'mexico': [-108.984, -92.434, 12, 23.021],
     'japan': [120, 150, 22, 47],
    'chile': [-75, -66, -30, -17],
    'alaska': [-170, -137, 50, 65], # 50.466,-170.089, 65.453,-137.684
    'cascadia': [-129, -119, 38, 51]  # 38.010,-129.542, 50.596,-119.421

}
fig_sizes = {
    'mexico': (10, 8),
             'japan': (10, 10),
    'chile': (6, 11),
    'alaska': (11, 6),
    'cascadia': (10, 10),

}


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
        plot_cbar = True if i == slabs[-1] else False
        plot_region_border = True if region != 'alaska' else False
        print(plot_region_border)
        ax = csep.utils.plots.plot_gridded_dataset(csep_region.get_cartesian(data), csep_region,
                                                   ax=ax, colorbar=plot_cbar,
                                                   plot_region=plot_region_border,
                                                   colormap='magma', alpha=0.5, clim=[0, 300], clabel='Depth [km]')

    ### Slab poly
    gdf = gpd.read_file(slabpoly_per_region[region])
    polygon = gdf.geometry.iloc[0]
    coords = np.array((polygon.exterior.coords))
    ax.plot(coords[:, 0], coords[:, 1], '-', color='gold')

    ### Fast Catalog
    catalogs[region][0].plot(ax=ax, markercolor='steelblue',
                             markeredgecolor='darkblue', size=1, power=2)


    ### Slowslip
    get_slow_slip_catalog().plot(ax=ax, markercolor='red',
                                size=1, power=2,
                                 min_val=catalogs[region][0].get_magnitudes().min(),
                                 max_val=catalogs[region][0].get_magnitudes().max(),
                                 mag_ticks=np.arange(np.floor(catalogs[region][0].get_magnitudes().min()),
                                                     np.ceil(catalogs[region][0].get_magnitudes().max()))
                                 )

    custom_legend = [
        Line2D([0], [0], marker='o', color='blue', markersize=8, label='Earthquakes',
               linestyle='None'),
        Line2D([0], [0], marker='o', color='red', markersize=8, label='Slow Slip Events',
               linestyle='None'),
        Line2D([0], [0], color='gold', linewidth=2, label='Filter Polygon'),
        Patch(facecolor='purple', label='Subduction surface', edgecolor='black'),
    ]

    # Add the legend to the existing ax without removing the current one
    first_legend = ax.get_legend()
    ncol = 4
    if region == 'chile':
        ncol = 2
    second_legend = ax.legend(handles=custom_legend, loc='lower center',
                              bbox_to_anchor=(0.5, 1.02), ncol=ncol,
              frameon=False)
    ax.coastlines(
        color='black', linewidth=1.5
    )
    ax.add_artist(first_legend)
    ax.add_artist(second_legend)
    plt.savefig(filename[region])





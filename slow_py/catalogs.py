import csep
import numpy as np
import os
import geopandas as gpd
import pandas
from shapely.geometry import Point
import matplotlib.pyplot as plt


basepath = os.path.dirname(__file__)


def get_chile_catalog(depth=None, savepath=None):

    if depth == '70km' or depth == 70:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_70km/chile.shp')
    elif depth == '150km' or depth == 150:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_150km/chile.shp')
    else:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_full/chile.shp')

    cat_fname =  os.path.join(basepath,'../data/fast_catalogue_CH_alldepths')

    catalog = csep.load_catalog(cat_fname)

    point_geometries = [Point(lon, lat) for lon, lat in zip(catalog.get_longitudes(),
                                                            catalog.get_latitudes())]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")
    region = gpd.read_file(shp_name)

    within_mask = points_gdf.within(region.union_all())

    new_data = catalog.data[within_mask]

    new_catalog = csep.core.catalogs.CSEPCatalog(data=new_data)

    if savepath:
        new_catalog.write_ascii(savepath)
    return new_catalog


def get_mexico_raw_catalog(depth=None, savepath=None):

    if depth == '70km' or depth == 70:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_70km/mexico.shp')
    elif depth == '150km' or depth == 150:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_150km/mexico.shp')
    else:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_full/mexico.shp')

    cat_fname =  os.path.join(basepath, '../data/fast_catalogue_MX_alldepths')

    catalog = csep.load_catalog(cat_fname)

    point_geometries = [Point(lon, lat) for lon, lat in zip(catalog.get_longitudes(),
                                                            catalog.get_latitudes())]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")
    region = gpd.read_file(shp_name)

    # within_mask = points_gdf.within(region.union_all())

    new_data = catalog.data

    new_catalog = csep.core.catalogs.CSEPCatalog(data=new_data)

    if savepath:
        new_catalog.write_ascii(savepath)
    return new_catalog


def get_mexico_catalog(depth=None, savepath=None):

    if depth == '70km' or depth == 70:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_70km/mexico.shp')
    elif depth == '150km' or depth == 150:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_150km/mexico.shp')
    else:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_full/mexico.shp')

    cat_fname =  os.path.join(basepath, '../data/fast_catalogue_MX_alldepths')

    catalog = csep.load_catalog(cat_fname)

    point_geometries = [Point(lon, lat) for lon, lat in zip(catalog.get_longitudes(),
                                                            catalog.get_latitudes())]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")
    region = gpd.read_file(shp_name)

    within_mask = points_gdf.within(region.union_all())

    new_data = catalog.data[within_mask]

    new_catalog = csep.core.catalogs.CSEPCatalog(data=new_data)

    if savepath:
        new_catalog.write_ascii(savepath)
    return new_catalog


def get_japan_catalog(depth=None, savepath=None):

    if depth == '70km' or depth == 70:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_70km/japan.shp')
    elif depth == '150km' or depth == 150:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_150km/japan.shp')
    else:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_full/japan.shp')

    cat_fname = '../data/Japan_EQ_all.csv'
    catalog = pandas.read_csv(cat_fname)

    # catalog = csep.load_catalog(cat_fname)

    point_geometries = [Point(lon, lat) for lon, lat in zip(catalog.longitude,
                                                            catalog.latitude)]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")
    region = gpd.read_file(shp_name)

    within_mask = points_gdf.within(region.union_all())

    new_catalog = catalog[within_mask]

    lons = new_catalog.longitude
    lats = new_catalog.latitude
    # csep.utils.time_utils.strptime_to_utc_epoch(catalog.datetime_utc[0])
    epoch_times = [csep.utils.time_utils.strptime_to_utc_epoch(i) for i in new_catalog.datetime_utc]
    depth = new_catalog.depth_km
    magnitude = new_catalog.magnitude

    events = []
    for i, (lon, lat, dt, depth, mag) in enumerate(zip(lons, lats, epoch_times, depth, magnitude)):
        events.append((i, dt, lat, lon, depth, mag))

    new_catalog = csep.core.catalogs.CSEPCatalog(data=events)
    if savepath:
        new_catalog.write_ascii(savepath)
    return new_catalog


def get_japan_filtered_catalog(depth=None, savepath=None):

    if depth == '70km' or depth == 70:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_70km/japan.shp')
    elif depth == '150km' or depth == 150:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_150km/japan.shp')
    else:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_full/japan.shp')

    cat_fname = '../data/Japan_EQ_filtered.csv'
    catalog = pandas.read_csv(cat_fname)

    # catalog = csep.load_catalog(cat_fname)

    point_geometries = [Point(lon, lat) for lon, lat in zip(catalog.longitude,
                                                            catalog.latitude)]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")
    region = gpd.read_file(shp_name)

    within_mask = points_gdf.within(region.union_all())

    new_catalog = catalog[within_mask]

    lons = new_catalog.longitude
    lats = new_catalog.latitude
    # csep.utils.time_utils.strptime_to_utc_epoch(catalog.datetime_utc[0])
    epoch_times = [csep.utils.time_utils.strptime_to_utc_epoch(i) for i in new_catalog.datetime_utc]
    depth = new_catalog.depth_km
    magnitude = new_catalog.magnitude

    events = []
    for i, (lon, lat, dt, depth, mag) in enumerate(zip(lons, lats, epoch_times, depth, magnitude)):
        events.append((i, dt, lat, lon, depth, mag))

    new_catalog = csep.core.catalogs.CSEPCatalog(data=events)
    if savepath:
        new_catalog.write_ascii(savepath)
    return new_catalog


def get_cascadia_catalog(depth=None, savepath=None):

    if depth == '70km' or depth == 70:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_70km/cas.shp')
    elif depth == '150km' or depth == 150:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_150km/cas.shp')
    else:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_full/cas.shp')

    cat_fname = '../data/Cascadia_EQ_all.csv'
    catalog = pandas.read_csv(cat_fname)

    # catalog = csep.load_catalog(cat_fname)

    point_geometries = [Point(lon, lat) for lon, lat in zip(catalog.longitude,
                                                            catalog.latitude)]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")
    region = gpd.read_file(shp_name)

    within_mask = points_gdf.within(region.union_all())

    new_catalog = catalog[within_mask]

    lons = new_catalog.longitude
    lats = new_catalog.latitude
    # csep.utils.time_utils.strptime_to_utc_epoch(catalog.datetime_utc[0])
    epoch_times = [csep.utils.time_utils.strptime_to_utc_epoch(i.replace('T', ' ').replace('Z', '')) for i in new_catalog.time]
    depth = new_catalog.depth
    magnitude = new_catalog.mag

    events = []
    for i, (lon, lat, dt, depth, mag) in enumerate(zip(lons, lats, epoch_times, depth, magnitude)):
        events.append((i, dt, lat, lon, depth, mag))

    new_catalog = csep.core.catalogs.CSEPCatalog(data=events)
    if savepath:
        new_catalog.write_ascii(savepath)
    return new_catalog

def get_alaska_catalog(depth=None, savepath=None):

    if depth == '70km' or depth == 70:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_70km/alu.shp')
    elif depth == '150km' or depth == 150:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_150km/alu.shp')
    else:
        shp_name = os.path.join(basepath,
                                '../data/poly4seismicity/FinalPolygons_full/alu.shp')

    cat_fname = '../data/Alaska_EQ_all.csv'
    catalog = pandas.read_csv(cat_fname)

    # catalog = csep.load_catalog(cat_fname)

    point_geometries = [Point(lon, lat) for lon, lat in zip(catalog.longitude,
                                                            catalog.latitude)]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")
    region = gpd.read_file(shp_name)

    within_mask = points_gdf.within(region.union_all())

    new_catalog = catalog[within_mask]

    lons = new_catalog.longitude
    lats = new_catalog.latitude
    # csep.utils.time_utils.strptime_to_utc_epoch(catalog.datetime_utc[0])
    epoch_times = [csep.utils.time_utils.strptime_to_utc_epoch(i.replace('T', ' ').replace('Z', '')) for i in new_catalog.time]
    depth = new_catalog.depth
    magnitude = new_catalog.mag

    events = []
    for i, (lon, lat, dt, depth, mag) in enumerate(zip(lons, lats, epoch_times, depth, magnitude)):
        events.append((i, dt, lat, lon, depth, mag))

    new_catalog = csep.core.catalogs.CSEPCatalog(data=events)
    if savepath:
        new_catalog.write_ascii(savepath)
    return new_catalog



def get_slow_slip_catalog():
    cat_fname = '../data/SSE_full_cat_pref_largest.csv'
    db = pandas.read_csv(cat_fname, index_col=0)

    lons = db.lon
    lats = db.lat
    # csep.utils.time_utils.strptime_to_utc_epoch(catalog.datetime_utc[0])
    epoch_times = [csep.utils.time_utils.strptime_to_utc_epoch(i.replace('T', ' ').replace('Z', '')) for i in db.time]
    depth = db.depth
    magnitude = db.mag

    events = []
    for i, (lon, lat, dt, depth, mag) in enumerate(zip(lons, lats, epoch_times, depth, magnitude)):
        events.append((i, dt, lat, lon, depth, mag))
    new_catalog = csep.core.catalogs.CSEPCatalog(data=events)
    return new_catalog

FUNC_MAP = {
    # 'chile': get_chile_catalog,
    # 'mexico': get_mexico_catalog,
    # 'japan': get_japan_catalog,
    # 'japan_filtered': get_japan_filtered_catalog,
    'cascadia': get_cascadia_catalog

}

if __name__ == '__main__':
    # savedir = os.path.join(basepath, '../data/FastEarthquakes')
    # plot = True
    # for cat_name in FUNC_MAP.keys():
    #     for depth in ['70km', '150km', 'full']:
    #         print(cat_name, depth)
    #         savepath = os.path.join(savedir, f'depth_{depth}', f'{cat_name}.csv')
    #         catalog = FUNC_MAP[cat_name](depth=depth, savepath=savepath)
    #         if plot:
    #             figpath = os.path.join(savedir, f'depth_{depth}', f'{cat_name}.png')
    #             catalog.plot()
    #             plt.savefig(figpath, dpi=200)

    a = get_slow_slip_catalog()




import numpy as np
import pandas as pd
import rasterio
import alphashape
from rasterio.transform import from_origin
from shapely import MultiPolygon, Polygon
from sklearn.neighbors import BallTree
from pathlib import Path
import utm
import warnings
from typing import Union, Literal
from functools import cached_property
from pyproj import Transformer
import geopandas as gpd



base_dir = Path(__file__).parents[2]
DATA_DIR = base_dir / "Datasets" / "Slab2" / "Slab2_TXT"

ALL_SLABS = {
    "cal": "Calabria",
    "cam": "Central_America",
    "cot": "Cotabato",
    "hin": "Hindu_Kush",
    "man": "Manila",
    "sco": "Scotia",
    "sul": "Sulawesi",
    "sam": "South_America",
    "cas": "Cascadia",
    "him": "Himalaya",
    "puy": "Puysegur",
    "mak": "Makran",
    "hal": "Halmahera",
    "kur": "Kuril",
    "mue": "Muertos",
    "alu": "Aleutian",
    "ryu": "Ryukyu",
    "phi": "Philippines",
    "ker": "Kermadec",
    "van": "Vanuatu",
    "png": "New_Guinea",
    "car": "Caribbean",
    "hel": "Hellenic",
    "pam": "Pamir",
    "sol": "Solomon",
    "sum": "Sumatra",
    "izu": "Izu_Bonin",
}

SLAB_PROPERTIES = {
    "dep": "depth",
    "dip": "dip",
    "str": "strike",
    "thk": "thickness",
    "unc": "uncertainty",
}

SLAB_MODEL_DATE = [
    "02.23.18",
    "02.24.18",
    "02.26.18",
]


class Slab:
    def __init__(
        self,
        name: str,
        path: Union[Path, str] = DATA_DIR,
        date: Union[list, str] = SLAB_MODEL_DATE,
    ):
        assert name in ALL_SLABS.keys(), f"Slab name {name} not in {ALL_SLABS}"

        self.name = name
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.date = date

        self.raw_xyz = self._get_property_xyz("dep")

        self.indices = np.where(~np.isnan(self.raw_xyz[:, 2]))

        self.longitude = np.where(
            self.raw_xyz[:, 0] > 180, self.raw_xyz[:, 0] - 360, self.raw_xyz[:, 0]
        )
        self.latitude = self.raw_xyz[:, 1]
        self.depth = -self.raw_xyz[:, 2] * 1000

        self.easting, self.northing, self.utm_zone, self.utm_letter = self.force_ll2utm(
            self.latitude, self.longitude
        )

    def _get_property_xyz(self, property: str):
        if type(self.date) == list:
            for idate in self.date:
                file = self.path / f"{self.name}_slab2_{property}_{idate}.xyz"
                if file.exists():
                    break
        else:
            file = self.path / f"{self.name}_slab2_{property}_{self.date}.xyz"
            assert file.exists()

        xyz = np.genfromtxt(
            file,
            delimiter=",",
            dtype=float,
            missing_values="NaN",
            filling_values=np.nan,
        )

        return xyz

    @cached_property
    def strike(self):
        return self._get_property_xyz("str")[:, 2]

    @cached_property
    def dip(self):
        return self._get_property_xyz("dip")[:, 2]

    @cached_property
    def thickness(self):
        return self._get_property_xyz("thk")[:, 2]

    @cached_property
    def uncertainty(self):
        return self._get_property_xyz("unc")[:, 2]

    def densify(self, step_meter: float = 1000):
        """Densify the slab by linear interpolation between coordinates of the geometry."""

        raise NotImplementedError("densify() not implemented yet")

    def distance(
        self,
        xyz,
        from_latlon: bool = True,  # else from ECEF
        depth_unit: Literal["km", "m"] = "m",
        distance_unit: Literal["km", "m"] = "m",
    ):
        """Calculates the distance between each point in xyz and the nearest point in the slab."""

        xyz = np.atleast_2d(xyz)
        assert xyz.shape[1] == 3, "xyz must have 3 columns"

        if np.any(xyz[:, 2] < 0):
            warnings.warn("xyz contains negative depths")
        if depth_unit == "km":
            xyz[:, 2] = xyz[:, 2] * 1000

        if from_latlon:
            ECEF_xyz = np.atleast_2d(
                np.column_stack(
                    self.gps_to_ecef_pyproj(xyz[:, 0], xyz[:, 1], -xyz[:, 2])
                )
            )
        else:
            ECEF_xyz = xyz

        slab_ECEF_xyz = np.array(
            self.gps_to_ecef_pyproj(self.latitude, self.longitude, -self.depth)
        ).T
        slab_ECEF_xyz = slab_ECEF_xyz[~np.isnan(slab_ECEF_xyz[:, 2]), :]
        tree = BallTree(slab_ECEF_xyz)

        query = tree.query(ECEF_xyz, return_distance=True)[
            0
        ].squeeze()  # [0] is the distance [1] is index

        if distance_unit == "km":
            query /= 1000.0

        return query

    def interpolate(
        self, property: str, lat: np.ndarray, lon: np.ndarray
    ):  # TODO: use ECEF instead of lat/lon
        """Interpolates the querried property at the given lat, lon using a nearest neighbor search.
        Assumes that the queried location is within the slab geometry.

        Args:
            property: property specified in SLAB_PROPERTIES
            lat: latitudes
            lon: longitudes

        Returns:
            interpolated property
        """
        mask = ~np.isnan(getattr(self, property))
        slab_xyz = np.array(
            [v[mask] for v in [self.latitude, self.longitude, getattr(self, property)]]
        ).T
        tree = BallTree(np.deg2rad(slab_xyz[:, :2]), metric="haversine")

        query = tree.query(
            np.deg2rad(np.column_stack([lat, lon])),
            return_distance=True,
        )[
            1
        ].squeeze()  # [0] is the distance [1] is index

        property_values = slab_xyz[query, 2]

        return property_values

    @staticmethod
    def force_ll2utm(
        lat, lon, force_zone_number=False, force_zone_letter=False
    ) -> tuple:
        # Hack to force utm to use the same zone for all points
        # (note I posted a stackoverflow question about this)
        if not force_zone_letter or not force_zone_letter:
            _, _, zone, letter = utm.from_latlon(np.mean(lat), np.mean(lon))
        else:
            zone, letter = [force_zone_number, force_zone_letter]
        I_positives = lat >= 0

        if np.sum(~I_positives) > 0 and np.sum(I_positives) > 0:
            east_pos, north_pos, _, _ = utm.from_latlon(
                lat[I_positives],
                lon[I_positives],
                force_zone_number=zone,
                force_zone_letter=letter,
            )
            east_neg, north_neg, _, _ = utm.from_latlon(
                lat[~I_positives],
                lon[~I_positives],
                force_zone_number=zone,
                force_zone_letter=letter,
            )

            east = np.concatenate((east_pos, east_neg))
            north = np.concatenate((north_pos, north_neg - 10000000))
        else:
            east, north, _, _ = utm.from_latlon(
                lat, lon, force_zone_number=zone, force_zone_letter=letter
            )
        return east, north, zone, letter

    @staticmethod
    def gps_to_ecef_pyproj(lat, lon, alt):
        transformer = Transformer.from_crs(
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        )
        return transformer.transform(lon, lat, alt, radians=False)

    def write_csv(self, path, attrs=['depth']):

        df = pd.DataFrame({"lon": self.longitude.ravel()[self.indices],
                           "lat": self.latitude.ravel()[self.indices]})
        for key in attrs:
            df[key] = getattr(self, key)[self.indices]
        df.to_csv(path, index=False)

    def write_geotiff(self, path, attr, corr360=False):

        if corr360:
            longitudes = self.raw_xyz[:, 0]
            latitudes = self.raw_xyz[:, 1]
        else:
            longitudes = self.longitude
            latitudes = self.latitude

        lon_min = np.min(longitudes)
        lon_max = np.max(longitudes)
        lat_min = np.min(latitudes)
        lat_max = np.max(latitudes)
        dx = np.diff(np.unique(longitudes))[0]
        dy = np.diff(np.unique(latitudes))[0]

        nx = int(np.round((lon_max - lon_min) / dx, 0)) + 1
        ny = int(np.round((lat_max - lat_min) / dy, 0)) + 1

        array = self.__dict__[attr].reshape(ny, nx)

        transform = from_origin(lon_min, lat_max, dx, dy)
        dtype = array.dtype
        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=dtype,
            crs="EPSG:4326",
            transform=transform
        ) as dst:
            dst.write(array, 1)

        #
        # lon_min, lon_max, lat_min, lat_max, res = parse_gmt_history(ds.attrs["history"])

    def write_shp(self, path, corr360=False):
        if corr360:
            longitudes = self.raw_xyz[:, 0]
            latitudes = self.raw_xyz[:, 1]
        else:
            longitudes = self.longitude
            latitudes = self.latitude


        # Estimate cell size (assumes regular grid!)
        dlon = np.diff(np.unique(longitudes))[0]
        dlat = np.diff(np.unique(latitudes))[0]

        # Build point set
        points = [(lon, lat) for lon, lat in zip(longitudes[self.indices], latitudes[self.indices])]

        # Compute alpha shape (alpha ≈ resolution works well for structured grids)
        alpha = dlat * 250
        print(f'res: {dlat}, alpha: {alpha}')
        hull = alphashape.alphashape(np.array(points), alpha=alpha)
        # Force the geometry to be a MultiPolygon if needed
        if isinstance(hull, (Polygon, MultiPolygon)):
            geom = hull
        elif hasattr(hull, "geoms"):
            print(hull)
            polys = [g for g in hull.geoms if isinstance(g, (Polygon, MultiPolygon))]
            geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
        else:
            raise ValueError("Resulting alpha shape is not a polygonal geometry.")
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

        # Export to shapefile
        gdf.to_file(path)


for slab_name in ALL_SLABS:
    print(slab_name)
    if slab_name != 'ker':
        continue
    slab = Slab(slab_name, path='../data/Slab2/xyz')
    slab.write_csv(f'../data/Slab2/csv/{slab_name}_slab2.csv')
    # slab.write_geotiff(f'../data/Slab2/geotiff/{slab_name}_depth.tif', attr='depth', corr360=True)
    slab.write_shp(f'../data/Slab2/shp/{slab_name}.shp', corr360=True)

# def write_singleband_tiff(path, array, transform, dtype):
#     array = array[::-1, :]  # Flip for north-up
#     with rasterio.open(
#         path, "w",
#         driver="GTiff",
#         height=array.shape[0],
#         width=array.shape[1],
#         count=1,
#         dtype=dtype,
#         crs="EPSG:4326",
#         transform=transform
#     ) as dst:
#         dst.write(array, 1)
#
#
# def write_csv(path, x, y, data):
#     xx, yy = np.meshgrid(x, y)
#     df = pd.DataFrame({"lon": xx.ravel(), "lat": yy.ravel()})
#     for key, arr in data.items():
#         df[key] = arr.ravel()
#     df.dropna(inplace=True)
#     df.to_csv(path, index=False)
#
#
# def write_vtk(path, x, y, data_dict, depth_scale=1.0, center_longitudes=False):
#     if "depth" not in data_dict:
#         raise ValueError("VTK export requires 'depth' in data_dict")
#
#     depth = data_dict["depth"]
#     ny, nx = depth.shape
#
#     xx, yy = np.meshgrid(x, y, indexing="xy")  # shape (ny, nx)
#
#     if center_longitudes:
#         xx = np.where(xx < 0, xx + 360, xx)
#
#         sort_idx = np.argsort(xx[0, :])
#         xx = xx[:, sort_idx]
#         yy = yy[:, sort_idx]
#         for key in data_dict:
#             data_dict[key] = data_dict[key][:, sort_idx]
#
#     zz = data_dict["depth"] * depth_scale
#     points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
#
#     grid = pv.StructuredGrid()
#     grid.points = points
#     grid.dimensions = [nx, ny, 1]
#
#     for name, arr in data_dict.items():
#         grid[name] = arr.ravel(order="C")
#
#     grid.save(path)
#
#
# def process_subduction(prefix, input_dir, output_dir):
#     print(f"Processing {prefix}...")
#
#     os.makedirs(f"{output_dir}/tiff", exist_ok=True)
#     os.makedirs(f"{output_dir}/csv", exist_ok=True)
#     os.makedirs(f"{output_dir}/vtk", exist_ok=True)
#
#     data = {}
#     shape = None
#     lon_min = lon_max = lat_min = lat_max = res = None
#
#     for suffix, varname in FILENAME_MAP.items():
#         file_path = os.path.join(input_dir, f"{prefix}_{suffix}")
#         if not os.path.exists(file_path):
#             continue
#         ds = xr.open_dataset(file_path)
#         z = ds["z"].values
#         data[varname] = z
#         if shape is None:
#             shape = z.shape
#             lon_min, lon_max, lat_min, lat_max, res = parse_gmt_history(ds.attrs["history"])
#
#     if not data:
#         print(f"No matching .grd files for {prefix}")
#         return
#
#     ny, nx = shape
#     # --- Handle longitude wrapping ---
#     if lon_min < 0: lon_min += 360
#     if lon_max < 0: lon_max += 360
#     if lon_max < lon_min:
#         lon_max += 360  # ensure wraparound is properly handled
#
#     x = np.linspace(lon_min, lon_max, nx)
#     y = np.linspace(lat_min, lat_max, ny)
#
#     # --- Adjust for Pacific-centered longitudes in raster and CSV ---
#     if lon_min < 0:
#         x = np.where(x < 0, x + 360, x)
#         sort_idx = np.argsort(x)
#         x = x[sort_idx]
#
#         for key in data:
#             data[key] = data[key][:, sort_idx]
#
#         lon_min = x[0]  # Update transform origin
#
#     transform = from_origin(lon_min, lat_max, res, res)
#
#     # Write TIFFs
#     for key, arr in data.items():
#         tif_path = f"{output_dir}/tiff/{prefix}_{key}.tif"
#         write_singleband_tiff(tif_path, arr, transform, arr.dtype)
#
#     # Write CSV
#     csv_path = f"{output_dir}/csv/{prefix}.csv"
#     write_csv(csv_path, x, y, data)
#
#     # Write VTK
#     vtk_path = f"{output_dir}/vtk/{prefix}.vtk"
#     write_vtk(
#         vtk_path,
#         x,
#         y,
#         data,
#         depth_scale=0.01,
#         center_longitudes=True
#     )
#
#
# # --- Run all prefixes ---
# input_dir = "../slab2_data/raw"
# output_dir = "../slab2_data/processed"
# prefixes = [
#     "cam_slab2_surf", "sam_slab2_center", "kur_slab2_surf",
#     "izu_slab2_surf", "cas_slab2_surf", "alu_slab2_surf"
# ]
#
# for prefix in prefixes:
#     process_subduction(prefix, input_dir, output_dir)

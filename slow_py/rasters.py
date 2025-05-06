import rasterio
from rasterio.transform import from_origin
import numpy as np

input_path = "../data/Basemaps/ESRI_basemap.tiff"

output_path = "../data/Basemaps/ESRI_basemap_pacific_centered.tif"
with rasterio.open(input_path) as src:
    data = src.read()  # shape: (4, height, width)
    profile = src.profile.copy()
    transform = src.transform

    height, width = src.height, src.width
    lon_min = transform.c
    lon_res = transform.a
    lat_res = transform.e

    print("Transform:", transform)
    print("Width:", width, "Height:", height)

    # Sanity check: raster must start at -180 and be aligned left-to-right
    assert np.isclose(lon_min, -180), "Expected raster to start at -180°"
    assert lon_res > 0, "Expected longitude to increase left to right"
    assert lat_res < 0, "Expected latitude to decrease top to bottom"

    # Drop 1 column if width is odd
    if width % 2 != 0:
        data = data[:, :, :-1]
        width -= 1

    mid = width // 2

    # Rearrange: move [-180, 0) to the right
    data_360 = np.concatenate((data[:, :, mid:], data[:, :, :mid]), axis=2)

    # Update transform to start at 0° instead of -180°
    new_transform = from_origin(0, transform.f, lon_res, -lat_res)

    # Update metadata
    profile.update({
        "transform": new_transform,
        "width": width,
        "height": height,
        "count": data.shape[0],
        "dtype": data.dtype,
        "compress": "lzw"
    })


    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data_360)




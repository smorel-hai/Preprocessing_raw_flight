import math
import requests
import os
import io
import re
import numpy as np
from PIL import Image, UnidentifiedImageError
from utils.oriented_bbox import get_bounding_box

# Geospatial libraries (Required for stitching)
import rasterio
from rasterio.transform import from_origin

# Disable DecompressionBombError for large satellite maps
Image.MAX_IMAGE_PIXELS = None

# --- Constants for Web Mercator (EPSG:3857) ---
R = 6378137
MAX_METERS = 2 * math.pi * R
ORIGIN_SHIFT = MAX_METERS / 2.0


def lat_lon_to_tile(lat, lon, zoom):
    """Calculates tile X, Y index for a given Lat/Lon and Zoom."""
    # Safety clamp for Web Mercator (Lat must be between -85.05 and 85.05)
    lat = max(min(lat, 85.0511), -85.0511)

    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = (my * ORIGIN_SHIFT) / 180.0

    res = MAX_METERS / (512 * (2 ** zoom))

    # Calculate global pixels
    px = (mx + ORIGIN_SHIFT) / res
    py = (ORIGIN_SHIFT - my) / res

    return int(px / 512), int(py / 512)


def get_tile_bounds_in_meters(tx, ty, zoom):
    """Returns the Web Mercator bounds of a specific tile (used for georeferencing)."""
    res = MAX_METERS / (512 * (2 ** zoom))
    min_x = (tx * 512 * res) - ORIGIN_SHIFT
    max_x = ((tx + 1) * 512 * res) - ORIGIN_SHIFT

    # In TMS/Google Grid, Y increases downwards, so max_y corresponds to ty (top of tile)
    max_y = ORIGIN_SHIFT - (ty * 512 * res)
    min_y = ORIGIN_SHIFT - ((ty + 1) * 512 * res)
    return min_x, min_y, max_x, max_y


def download_tiles(top_left, bottom_right, api_key, output_dir, zoom=19):
    """
    Downloads individual tiles within a bounding box at a specific zoom level.
    Saves them as individual JPG files.
    """
    # --- 1. Directory Setup ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- 2. Calculate Grid ---
    lat_max, lon_min = top_left
    lat_min, lon_max = bottom_right

    x_min, y_min = lat_lon_to_tile(lat_max, lon_min, zoom)
    x_max, y_max = lat_lon_to_tile(lat_min, lon_max, zoom)

    width_tiles = x_max - x_min + 1
    height_tiles = y_max - y_min + 1

    print(f"--- Processing Job ---")
    print(f"Region: {top_left} to {bottom_right}")
    print(f"Zoom Level: {zoom}")
    print(f"Grid: {width_tiles}x{height_tiles} tiles")
    print(f"Total tiles to download: {width_tiles * height_tiles}")

    # Headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    count = 0
    total = width_tiles * height_tiles

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://api.maptiler.com/tiles/satellite-v2/{zoom}/{x}/{y}.jpg?key={api_key}"
            filename = f"tile_z{zoom}_x{x}_y{y}.jpg"
            file_path = os.path.join(output_dir, filename)

            # Skip if already exists (optional, but good for large jobs)
            if os.path.exists(file_path):
                print(f"Skipping existing: {filename}", end='\r')
                count += 1
                continue

            try:
                r = requests.get(url, headers=headers, stream=True, timeout=10)

                if r.status_code == 200:
                    # Optional: Verify it's a valid image before saving
                    try:
                        # Just to check validity
                        Image.open(io.BytesIO(r.content))
                        with open(file_path, 'wb') as f:
                            f.write(r.content)
                    except UnidentifiedImageError:
                        print(
                            f"\nWarning: Tile {x},{y} returned 200 but wasn't a valid image.")
                else:
                    print(f"\nFailed: Tile {x},{y} Status: {r.status_code}")

                count += 1
                print(
                    f"Progress: [{count}/{total}] Downloaded {filename}", end='\r')

            except Exception as e:
                print(f"\nError downloading {x},{y}: {e}")

    print(f"\n✅ Success! All tiles saved in: {output_dir}")


def merge_tiles_to_geotiff(input_dir, output_file, zoom):
    """
    Reads all tiles in input_dir (matching the naming pattern and zoom), 
    stitches them, and saves a GeoTIFF.
    """
    print(f"\n--- Starting Stitching Process for Zoom {zoom} ---")

    # 1. Scan directory for valid tiles matching the specific ZOOM
    pattern = re.compile(r"tile_z(\d+)_x(\d+)_y(\d+)\.jpg")
    files = []

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    for f in os.listdir(input_dir):
        match = pattern.match(f)
        if match:
            z, x, y = map(int, match.groups())
            # IMPORTANT: Filter by zoom to avoid mixing different resolution tiles
            if z == zoom:
                files.append({'filename': f, 'z': z, 'x': x, 'y': y})

    if not files:
        print(f"No tile files found matching pattern 'tile_z{zoom}_x*_y*.jpg'")
        return

    # 2. Determine bounds
    min_x = min(f['x'] for f in files)
    max_x = max(f['x'] for f in files)
    min_y = min(f['y'] for f in files)
    max_y = max(f['y'] for f in files)

    width_tiles = max_x - min_x + 1
    height_tiles = max_y - min_y + 1

    full_w = width_tiles * 512
    full_h = height_tiles * 512

    print(full_w, full_h)

    print(
        f"Detected Mosaic Size: {width_tiles}x{height_tiles} tiles ({full_w}x{full_h} pixels)")

    # 3. Create Mosaic Canvas
    mosaic = Image.new('RGB', (full_w, full_h), (0, 0, 0))

    print("Stitching tiles...")
    for i, f in enumerate(files):
        try:
            # Calculate pixel position relative to top-left of the mosaic
            px = (f['x'] - min_x) * 512
            py = (f['y'] - min_y) * 512

            tile_path = os.path.join(input_dir, f['filename'])
            tile_img = Image.open(tile_path)
            mosaic.paste(tile_img, (px, py))
        except Exception as e:
            print(f"Error reading {f['filename']}: {e}")

    print("Stitching complete in memory.")

    # 4. Georeference
    # We need the Web Mercator bounds of the TOP-LEFT tile to anchor the map.
    # Note: min_y corresponds to the top row of tiles.
    min_mx, min_my, max_mx, max_my = get_tile_bounds_in_meters(
        min_x, min_y, zoom)

    # Pixel size in meters
    pixel_size = (max_mx - min_mx) / 512

    # Rasterio Transform: (West, North, pixel_width, pixel_height)
    # North bound is max_my of the top row.
    transform = from_origin(min_mx, max_my, pixel_size, pixel_size)

    # 5. Save to GeoTIFF
    img_array = np.array(mosaic)
    # Convert HWC to CHW for Rasterio
    img_array = np.moveaxis(img_array, -1, 0)

    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=img_array.shape[1],
        width=img_array.shape[2],
        count=3,
        dtype=img_array.dtype,
        crs='EPSG:3857',
        transform=transform,
        compress='lzw',
        nodata=0  # Makes black background transparent
    ) as dst:
        dst.write(img_array)

    print(f"✅ Created GeoTIFF: {output_file}")


# --- Usage Example ---
if __name__ == "__main__":
    # Your MapTiler Key
    MY_API_KEY = "SZ5Q6ilGzFm9Wge4GYp8"
    TARGET_DIRECTORY = "Downloaded_Tiles"
    OUTPUT_TIFF = "stitched_map.tif"

    # 2. Define Area (Lat, Lon)
    p1 = (48.702500, 2.024000)
    p2 = (48.698000, 2.029500)

    ZOOM_LEVEL = 19
    nw_corner, se_corner = get_bounding_box([p1, p2])

    # Step 1: Download
    download_tiles(
        nw_corner,
        se_corner,
        MY_API_KEY,
        output_dir=TARGET_DIRECTORY,
        zoom=ZOOM_LEVEL
    )

    # Step 2: Aggregate/Stitch
    merge_tiles_to_geotiff(TARGET_DIRECTORY, OUTPUT_TIFF, zoom=ZOOM_LEVEL)

"""Satellite tile extraction module for matching drone field-of-view.

This module extracts specific regions from large satellite GeoTIFF files
that correspond to drone camera field-of-view polygons.
"""

import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_geom
from shapely.geometry import Polygon


def get_best_tile_for_fov(tiff_path: str, fov_coords: list, src_crs: str = "EPSG:3857") -> dict:
    """Extract image data from a GeoTIFF that covers the given Field of View.

    Args:
        tiff_path: Path to the source GeoTIFF file
        fov_coords: List of (x, y) coordinates defining the FOV polygon
        src_crs: Coordinate reference system of fov_coords (default: Web Mercator)

    Returns:
        Dictionary containing:
            - image_data: Numpy array of the extracted tile
            - transform: Rasterio transform for the tile
            - crs: Coordinate reference system
            - coverage_ratio: Fraction of FOV covered by the TIFF
        Returns None if FOV is completely outside TIFF bounds
    """
    # 1. Create Shapely Polygon from FOV
    fov_poly = Polygon(fov_coords)
    if not fov_poly.is_valid:
        fov_poly = fov_poly.convex_hull

    with rasterio.open(tiff_path) as src:

        # 2. Handle Projection
        if src.crs.to_string() != src_crs:
            transformed_geom = transform_geom(
                src_crs,
                src.crs.to_string(),
                fov_poly.__geo_interface__
            )
            fov_poly_tiff_crs = Polygon(transformed_geom['coordinates'][0])
        else:
            fov_poly_tiff_crs = fov_poly

        # 3. Calculate the Bounding Box
        minx, miny, maxx, maxy = fov_poly_tiff_crs.bounds

        # 4. Generate the "Window" (With Integer Fix for safety)
        # We use round_offsets and round_shape to ensure we have valid integers for the intersection
        raw_window = from_bounds(
            minx, miny, maxx, maxy, transform=src.transform)
        safe_window = raw_window.round_offsets(
            op='floor').round_lengths(op='ceil')

        # Clip window to image bounds
        src_window = Window(0, 0, src.width, src.height)
        window = safe_window.intersection(src_window)

        # 5. Read the data
        if window.width <= 0 or window.height <= 0:
            print("Warning: FOV is completely outside the Tiff bounds.")
            return None

        tile_data = src.read(window=window)

        # Calculate new transform for this specific small crop
        tile_transform = src.window_transform(window)

        # Capture the CRS to pass it to the saving function
        file_crs = src.crs

        # 6. Calculate Coverage
        tiff_bounds_poly = Polygon([
            (src.bounds.left, src.bounds.bottom),
            (src.bounds.right, src.bounds.bottom),
            (src.bounds.right, src.bounds.top),
            (src.bounds.left, src.bounds.top)
        ])

        intersection = fov_poly_tiff_crs.intersection(tiff_bounds_poly).area
        coverage_ratio = intersection / fov_poly_tiff_crs.area

        return {
            "image_data": tile_data,
            "transform": tile_transform,
            "crs": file_crs,  # <--- NEW: We need this to save the file
            "coverage_ratio": coverage_ratio
        }


def save_tile_to_disk(result_dict: dict, output_filename: str) -> None:
    """Save extracted tile data to a GeoTIFF file.

    Args:
        result_dict: Dictionary from get_best_tile_for_fov containing:
            - image_data: Numpy array of pixel data
            - transform: Geospatial transform
            - crs: Coordinate reference system
        output_filename: Output file path (.tif/.tiff extension optional)
    """
    data = result_dict['image_data']
    transform = result_dict['transform']
    crs = result_dict['crs']

    # Get dimensions (Bands, Height, Width)
    count, height, width = data.shape

    # Ensure output has proper extension
    if not output_filename.lower().endswith(('.tif', '.tiff')):
        full_path = f"{output_filename}.tif"
    else:
        full_path = output_filename

    # Define metadata profile for the new GeoTIFF
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'count': count,
        'height': height,
        'width': width,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',  # Good for saving disk space
        'nodata': 0
    }

    # Write the file
    try:
        with rasterio.open(full_path, 'w', **profile) as dst:
            dst.write(data)
        print(f"✅ Saved successfully: {full_path}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

# --- Example Usage ---


if __name__ == "__main__":
    tiff_filename = "example_map.tif"

    # Define FOV
    fov = [
        (255000, 6250000),
        (256000, 6250000),
        (256000, 6251000),
        (255000, 6251000)
    ]

    # --- 1. Your Variable Name Logic ---
    # This is the variable that determines the file name
    desired_image_name = "Paris_FOV_042"

    try:
        # --- 2. Extract ---
        result = get_best_tile_for_fov(tiff_filename, fov)

        if result:
            print(f"Extracted shape: {result['image_data'].shape}")

            # --- 3. Save using the variable ---
            save_tile_to_disk(result, desired_image_name)

    except rasterio.errors.RasterioIOError:
        print("Input file not found. Please provide a real .tif file path.")

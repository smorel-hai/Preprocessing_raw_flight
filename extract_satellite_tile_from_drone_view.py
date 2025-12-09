import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_geom
from shapely.geometry import Polygon


def get_best_tile_for_fov(tiff_path, fov_coords, src_crs="EPSG:3857"):
    """
    Extracts the image data (tile) from a TIFF that covers the given Field of View.
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


def save_tile_to_disk(result_dict, output_filename_variable):
    """
    Saves the extracted tile data to a GeoTIFF file.

    Args:
        result_dict (dict): The output from get_best_tile_for_fov
        output_filename_variable (str): The name you want for the file (e.g. "image_01")
    """
    data = result_dict['image_data']
    transform = result_dict['transform']
    crs = result_dict['crs']

    # Get dimensions (Bands, Height, Width)
    count, height, width = data.shape

    # Define metadata profile for the new Tiff
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'count': count,
        'height': height,
        'width': width,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',  # Optional: Good for saving disk space
        'nodata': 0
    }

    # Ensure extension exists
    if not output_filename_variable.lower().endswith(('.tif', '.tiff')):
        full_path = f"{output_filename_variable}.tif"
    else:
        full_path = output_filename_variable

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

import json
import math
import os
from pathlib import Path


def _swap_coords(coords_list):
    """Converts [Lat, Lon] to [Lon, Lat] for GeoJSON."""
    return [[pt[1], pt[0]] for pt in coords_list]


def _ensure_closed(coords_list):
    """Closes the polygon loop."""
    if coords_list and coords_list[0] != coords_list[-1]:
        coords_list.append(coords_list[0])
    return coords_list


def _order_points_clockwise(points):
    """Sorts points clockwise around centroid to avoid bowtie shapes."""
    if not points:
        return points
    center_x = sum([p[0] for p in points]) / len(points)
    center_y = sum([p[1] for p in points]) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x))


def parse_data(root_dir: str, output_dir: str = None) -> dict:
    """
    Parses metadata.json and generates GeoJSON files for QGIS.
    """
    root_path = Path(root_dir)
    metadata_path = root_path / 'metadata.json'

    if output_dir is None:
        output_dir = root_path / 'qgis_data'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize Layers
    layers = {
        "regions": {"type": "FeatureCollection", "features": []},
        "zones": {"type": "FeatureCollection", "features": []},
        "drone_fov": {"type": "FeatureCollection", "features": []},
        "drone_pos": {"type": "FeatureCollection", "features": []},
        "drone_lines": {"type": "FeatureCollection", "features": []}
    }

    # Load Metadata
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    print("Parsing metadata...")

    for region_name, region_data in data.items():
        if region_name == 'devices_intrinsec_settings':
            continue

        # --- 1. REGION ---
        region_raw = _swap_coords(region_data['bbox_wgs84'])
        region_ordered = _order_points_clockwise(region_raw)
        layers['regions']['features'].append({
            "type": "Feature",
            "properties": {"name": region_name},
            "geometry": {"type": "Polygon", "coordinates": [_ensure_closed(region_ordered)]}
        })

        # --- 2. ZONES ---
        for zone_name, zone_data in region_data.get('zones', {}).items():

            # Link Tiff Paths
            valid_tiff_paths = []
            satellite_dict = zone_data.get('satellite', {})
            for sat_key, sat_rel_path in satellite_dict.items():
                tiff_path = root_path / region_name / sat_rel_path
                if tiff_path.exists():
                    valid_tiff_paths.append(str(sat_rel_path))

            zone_raw = _swap_coords(zone_data['bbox_wgs84'])
            zone_ordered = _order_points_clockwise(zone_raw)

            layers['zones']['features'].append({
                "type": "Feature",
                "properties": {
                    "name": zone_name,
                    "region": region_name,
                    "tiff_files": "; ".join(valid_tiff_paths)
                },
                "geometry": {"type": "Polygon", "coordinates": [_ensure_closed(zone_ordered)]}
            })

            # --- 3. DRONE DATA ---
            for frame_id, frame_data in zone_data.get('drone', {}).items():
                lat = frame_data.get('Lattitude') or frame_data.get('Latitude')
                lon = frame_data.get('Longitude')
                alt = frame_data.get('Absolute Altitude', 0)
                fov_raw = frame_data.get('fov_wgs84', [])

                if lat is not None and lon is not None:
                    cam_pt = [float(lon), float(lat)]

                    # Prepare Common Properties (This is the key for filtering!)
                    common_props = {
                        "id": frame_id,
                        "zone": zone_name,        # <--- ADDED for Filtering
                        "region": region_name     # <--- ADDED for Filtering
                    }

                    # Position Point
                    pos_props = common_props.copy()
                    pos_props["altitude"] = alt

                    layers['drone_pos']['features'].append({
                        "type": "Feature",
                        "properties": pos_props,
                        "geometry": {"type": "Point", "coordinates": cam_pt}
                    })

                    if fov_raw:
                        fov_swapped = _swap_coords(fov_raw)
                        fov_ordered = _order_points_clockwise(fov_swapped)

                        # FOV Polygon
                        layers['drone_fov']['features'].append({
                            "type": "Feature",
                            "properties": common_props,  # Inherits zone/region
                            "geometry": {"type": "Polygon", "coordinates": [_ensure_closed(fov_ordered.copy())]}
                        })

                        # Projection Lines
                        lines = [[cam_pt, corner] for corner in fov_ordered]
                        layers['drone_lines']['features'].append({
                            "type": "Feature",
                            "properties": common_props,  # Inherits zone/region
                            "geometry": {"type": "MultiLineString", "coordinates": lines}
                        })

    # Save to Disk
    generated_files = {}
    for layer_name, geojson_data in layers.items():
        out_path = output_dir / f"{layer_name}.geojson"
        with open(out_path, 'w') as f:
            json.dump(geojson_data, f)
        generated_files[layer_name] = str(out_path)

    print(f"GeoJSONs saved to {output_dir}")
    return generated_files


if __name__ == '__main__':
    ROOT_DIR = 'data/preprocessing_extraction'
    OUTPUT_DIR = None  # Defaults to 'qgis_data' inside ROOT_DIR
    parse_data(ROOT_DIR, OUTPUT_DIR)

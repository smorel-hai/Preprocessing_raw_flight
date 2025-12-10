import json
import math
from preprocessing_rawflight_pipeline.utils.utils import convert_mercator_to_wgs84

# --- Predefined symbols supported by most GeoJSON viewers (SimpleStyle) ---
# We will cycle through these for different cameras
AVAILABLE_SYMBOLS = [
    "circle", "square", "triangle", "rocket", "marker",
    "harbor", "airport", "bus", "rail-metro"
]


def calculate_haversine_distance(coord1, coord2):
    """Calculates distance in meters between two WGS84 points."""
    R = 6371000
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def reorder_corners(points):
    """ Sorts points counter-clockwise around their centroid. """
    if not points:
        return []
    center_x = sum([p[0] for p in points]) / len(points)
    center_y = sum([p[1] for p in points]) / len(points)

    def get_angle(point):
        return math.atan2(point[1] - center_y, point[0] - center_x)

    return sorted(points, key=get_angle)


def generate_position_comparison_geojson(pairs_list, output_file="position_errors.geojson"):
    features = []

    for item in pairs_list:
        pair_id = item.get("id", "unknown")
        camera_name = item.get("camera_name", "Unknown_Camera")
        real_pos = item["real"]
        est_pos = item["estimated"]
        error_meters = calculate_haversine_distance(real_pos, est_pos)

        base_props = {
            "pair_id": pair_id,
            "camera_setup": camera_name,
            "error_meters": round(error_meters, 2)
        }

        # --- 1. Real Position (Ground Truth) ---
        # Always a 'Star' to represent the target/truth
        real_feature = {
            "type": "Feature",
            "properties": {
                **base_props,
                "type": "real_position",
                "label": f"{pair_id} (Real)",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [real_pos[1], real_pos[0]]
            }
        }

        # --- 2. Estimated Position ---
        # Icon changes based on the camera
        est_feature = {
            "type": "Feature",
            "properties": {
                **base_props,
                "type": "estimated_position",
                "label": f"{pair_id} (Est)",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [est_pos[1], est_pos[0]]
            }
        }

        # --- 3. Connection Line ---
        line_feature = {
            "type": "Feature",
            "properties": {
                **base_props,
                "stroke-width": 2,
                "stroke-opacity": 0.6
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [real_pos[1], real_pos[0]],
                    [est_pos[1], est_pos[0]]
                ]
            }
        }

        features.extend([real_feature, est_feature, line_feature])

    geojson_struct = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_file, 'w') as f:
        json.dump(geojson_struct, f, indent=2)

    print(f"File '{output_file}' generated.")


def generate_multi_camera_geojson(camera_list, output_file="multi_camera_view"):
    """
    Generates a GeoJSON file for multiple camera setups.

    :param camera_list: A list of dictionaries. Each dict must have:
                        {'id': str, 'pos': (lat, lon), 'fov': [(x,y), ...]}
    """

    all_features = []

    for cam_data in camera_list:
        cam_id = cam_data.get("id", "unknown")
        cam_lat, cam_lon = cam_data["pos"]
        raw_fov = cam_data["fov"]

        # 1. Reorder and Convert FOV points
        sorted_fov_mercator = reorder_corners(raw_fov)
        fov_coords_wgs84 = convert_mercator_to_wgs84(sorted_fov_mercator, api_order=True)

        # Close the polygon loop
        if fov_coords_wgs84 and fov_coords_wgs84[0] != fov_coords_wgs84[-1]:
            fov_coords_wgs84.append(fov_coords_wgs84[0])

        # 2. Create Camera Feature (Point)
        camera_feature = {
            "type": "Feature",
            "properties": {
                "id": cam_id,
                "type": "camera",
                "description": f"Camera {cam_id}"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [cam_lon, cam_lat]
            }
        }

        # 3. Create FOV Feature (Polygon)
        fov_feature = {
            "type": "Feature",
            "properties": {
                "id": cam_id,
                "type": "fov",
                "parent_camera": cam_id
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [fov_coords_wgs84]
            }
        }

        # 4. Create Sight Lines (Visual Aid)
        # Connects camera to the corners to create a "cone" visual
        # We create a MultiLineString to connect the camera to every corner
        sight_lines_coords = []
        for corner in fov_coords_wgs84[:-1]:  # Skip the duplicate last point
            sight_lines_coords.append([[cam_lon, cam_lat], corner])

        line_feature = {
            "type": "Feature",
            "properties": {
                "id": cam_id,
                "type": "sightline",
                "parent_camera": cam_id
            },
            "geometry": {
                "type": "MultiLineString",
                "coordinates": sight_lines_coords
            }
        }

        # Add all to the master list
        all_features.extend([camera_feature, fov_feature, line_feature])

    # 5. Write Final File
    geojson_structure = {
        "type": "FeatureCollection",
        "features": all_features
    }

    with open(output_file, 'w') as f:
        json.dump(geojson_structure, f, indent=2)

    print(f"Successfully created '{output_file}' with {len(camera_list)} cameras.")


if __name__ == "__main__":

    # --- USAGE EXAMPLE ---

    data_pairs = [
        {
            "id": "Drone_01",
            "real": (48.8584, 2.2945),       # Eiffel Tower
            "estimated": (48.8585, 2.2946)   # Slightly off (approx 15m error)
        },
        {
            "id": "Drone_02",
            "real": (48.8606, 2.3376),       # Louvre
            "estimated": (48.8610, 2.3380)   # More off (approx 50m error)
        },
        {
            "id": "Drone_03",
            "real": (48.8529, 2.3499),       # Notre Dame
            "estimated": (48.8529, 2.3499)   # Perfect match (0m error)
        }
    ]

    generate_position_comparison_geojson(data_pairs)

import numpy as np
from shapely.geometry import Polygon


def get_forward_vector(rotation_matrix):
    """
    Extracts the forward viewing vector from a Rotation Matrix.
    Assumes standard convention where the camera looks down the Z-axis (local [0,0,1]).

    If your convention is different (e.g., Y-axis), change the index below.
    """
    matrix = np.array(rotation_matrix)
    # The 3rd column (index 2) usually represents the Z-axis vector in world coordinates
    forward_vector = matrix[:, 2]
    return forward_vector / np.linalg.norm(forward_vector)


def calculate_angle_degrees(vec1, vec2):
    """
    Calculates the angle in degrees between two normalized 3D vectors.
    """
    # Dot product: a . b = |a||b|cos(theta) -> since normalized: cos(theta)
    dot_product = np.dot(vec1, vec2)
    # Clip to handle floating point errors slightly outside [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def calculate_iou(poly1, poly2):
    if not poly1.intersects(poly2):
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

# --- MAIN FUNCTION ---


def prune_redundant_areas_with_rotation(
    coords_list,
    rotation_matrices,
    max_areas_to_keep,
    guide_coords_list=None,
    redundancy_iou_threshold=0.5,
    angle_threshold_degrees=15.0,
    min_inclusion_ratio=0.99  # 99% inside implies "Totally Inside" (allows for tiny rounding errors)
):
    """
    1. Guide Selection: Isolates items TOTALLY INSIDE the guides.
       - Keeps the biggest item per guide.
       - DISCARDS all other items that are totally inside guides.
    2. Standard Pruning: Runs on items that are OUTSIDE the guides.

    Returns:
    {
        "guided_matches": [ {'guide_index': 0, 'id': 12}, ... ],
        "standard_matches": [ 45, 98, ... ]
    }
    """

    # --- 1. Pre-process Candidates ---
    candidate_data = []
    for i, coords in enumerate(coords_list):
        try:
            p = Polygon(coords)
            if not p.is_valid:
                p = p.convex_hull
            if p.is_valid and p.area > 0:
                candidate_data.append({
                    'poly': p,
                    'vec': get_forward_vector(rotation_matrices[i]),
                    'id': i,
                    'area': p.area
                })
        except ValueError:
            continue

    # --- 2. Pre-process Guides ---
    guide_data = []
    if guide_coords_list:
        for i, g_coords in enumerate(guide_coords_list):
            try:
                p = Polygon(g_coords)
                if not p.is_valid:
                    p = p.convex_hull
                if p.is_valid and p.area > 0:
                    guide_data.append({'poly': p, 'index': i})
            except ValueError:
                continue

    all_kept_items = []
    guided_matches_output = {}

    # Track IDs that were found totally inside guides (Winners AND Losers)
    # These will be strictly excluded from Phase 2
    ids_found_totally_inside = set()

    # --- 3. PHASE 1: Guide Selection (Strict Inclusion) ---

    for guide in guide_data:
        if len(all_kept_items) >= max_areas_to_keep:
            break

        candidates_inside_this_guide = []

        for cand in candidate_data:
            # Quick check: does it intersect?
            if guide['poly'].intersects(cand['poly']):
                inter_area = guide['poly'].intersection(cand['poly']).area
                cand_area = cand['poly'].area

                if cand_area > 0:
                    ratio_inside = inter_area / cand_area

                    # CRITERIA: Strictly Inside (e.g., > 99%)
                    if ratio_inside >= min_inclusion_ratio:
                        candidates_inside_this_guide.append(cand)

                        # Mark this candidate as "Inside a Guide".
                        # It will be processed here (win or lose) and skipped in Phase 2.
                        ids_found_totally_inside.add(cand['id'])

        # Pick the winner for this guide (Biggest item totally inside)
        if candidates_inside_this_guide:
            candidates_inside_this_guide.sort(key=lambda x: x['area'], reverse=True)
            winner = candidates_inside_this_guide[0]

            guided_matches_output[winner['id']] = guide['index']

            # Add to main kept list if not duplicate
            if not any(k['id'] == winner['id'] for k in all_kept_items):
                all_kept_items.append(winner)

    # --- 4. PHASE 2: Standard Pruning for Outsiders ---

    # Outsider Pool = Candidates that were NOT totally inside any guide.
    # (Items that were partially inside or completely outside are allowed here)
    outsider_pool = [c for c in candidate_data if c['id'] not in ids_found_totally_inside]

    # Sort by Area (Descending)
    outsider_pool.sort(key=lambda x: x['area'], reverse=True)

    for current in outsider_pool:
        if len(all_kept_items) >= max_areas_to_keep:
            break

        is_redundant = False

        for kept in all_kept_items:
            # Check overlap against Winners from Phase 1 AND other Phase 2 selections
            iou = calculate_iou(current['poly'], kept['poly'])

            if iou > redundancy_iou_threshold:
                angle_diff = calculate_angle_degrees(current['vec'], kept['vec'])
                if angle_diff < angle_threshold_degrees:
                    is_redundant = True
                    break

        if not is_redundant:
            all_kept_items.append(current)
            guided_matches_output[current['id']] = None

    return guided_matches_output

# --- Example Usage ---


if __name__ == "__main__":
    # Helper to create a simple rotation matrix around Y axis (horizontal turn)
    def rotation_y(degrees):
        rad = np.radians(degrees)
        c, s = np.cos(rad), np.sin(rad)
        # Standard rotation matrix
        return [
            [c,  0, s],
            [0,  1, 0],
            [-s, 0, c]
        ]

    # --- Scenario ---
    # We have 3 identical squares (perfect overlap).
    # 1. Looking North (0 deg)
    # 2. Looking North (5 deg difference) -> Should be PRUNED (Redundant)
    # 3. Looking East (90 deg difference) -> Should be KEPT (Different view)

    square_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]

    coords = [square_coords, square_coords, square_coords]

    matrices = [
        rotation_y(0),   # Index 0: Base view
        rotation_y(5),   # Index 1: Very similar to 0 (Should be removed)
        rotation_y(90)   # Index 2: Perpendicular view (Should be kept)
    ]

    print(f"Total Inputs: {len(coords)}")

    # Run Pruning
    # angle_threshold=10 means views within 10 degrees of each other are redundant
    indices = prune_redundant_areas_with_rotation(
        coords,
        matrices,
        max_areas_to_keep=10,
        iou_threshold=0.5,
        angle_threshold_degrees=10.0
    )

    print(f"Kept Indices: {indices}")

    if 0 in indices and 2 in indices and 1 not in indices:
        print("SUCCESS: Kept distinct views, removed similar view.")
    else:
        print("FAIL: Logic incorrect.")

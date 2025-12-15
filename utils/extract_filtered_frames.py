import re
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
import cv2


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


DTYPES = {
    "Timestamp": str,
    "FrameNumber": int,
    "DiffTime": str,
    "Focal Length": float,
    "DZoom Ratio": float,
    "Latitude": float,
    "Longitude": float,
    "Relative Altitude": float,
    "Absolute Altitude": float,
    "Gimbal Yaw": float,
    "Gimbal Pitch": float,
    "Gimbal Roll": float,
}


class DJIMetadata:
    def __init__(self, metadata: List[Dict]):
        self.metadata = metadata
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        for frame_data in self.metadata:
            for key, value in frame_data.items():
                if key not in DTYPES:
                    continue
                if value is None:
                    continue
                try:
                    frame_data[key] = DTYPES[key](value)
                except ValueError:
                    pass


class DJIMetadataParser:
    def __call__(self, srt_path: str) -> DJIMetadata:
        return DJIMetadata(self._parse_srt(srt_path))

    def _parse_srt(self, file_path: str) -> List[Dict]:
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()

        self._split_by_frames(content)
        self.extracted_data = []

        for frame_metadata in self.frames_metadata:
            if frame_metadata.strip():
                self._parse_frame(frame_metadata)

        return self.extracted_data

    def _split_by_frames(self, content: str) -> None:
        self.frames_metadata = re.split(r"\n\s*\n", content.strip())

    def _parse_frame(self, frame_metadata: str) -> None:
        self.current_frame_metadata = {}
        self._parse_frame_metadata(frame_metadata)
        if self.current_frame_metadata:
            self._save_frame_metadata()

    def _parse_frame_metadata(self, frame_metadata: str) -> None:
        lines = frame_metadata.split("\n")
        if len(lines) >= 5:
            frame_info_line = lines[2]
            self.current_frame_metadata["FrameNumber"] = self._parse_single_metadata(
                r"FrameCnt: (\d+)", frame_info_line)
            self.current_frame_metadata["DiffTime"] = self._parse_single_metadata(
                r"DiffTime: (\d+ms)", frame_info_line)

            timestamp_line = lines[3]
            self.current_frame_metadata["Timestamp"] = self._parse_single_metadata(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", timestamp_line)

            metadata_line = lines[4]
            self.current_frame_metadata["Focal Length"] = self._parse_single_metadata(
                r"focal_len: ([\d.]+)", metadata_line)
            self.current_frame_metadata["DZoom Ratio"] = self._parse_single_metadata(
                r"dzoom_ratio: ([\d.]+)", metadata_line)
            self.current_frame_metadata["Latitude"] = self._parse_single_metadata(
                r"latitude: ([-\d.]+)", metadata_line)
            self.current_frame_metadata["Longitude"] = self._parse_single_metadata(
                r"longitude: ([-\d.]+)", metadata_line)
            self.current_frame_metadata["Relative Altitude"] = self._parse_single_metadata(
                r"rel_alt: ([\d.-]+)", metadata_line)
            self.current_frame_metadata["Absolute Altitude"] = self._parse_single_metadata(
                r"abs_alt: ([\d.-]+)", metadata_line)
            self.current_frame_metadata["Gimbal Yaw"] = self._parse_single_metadata(
                r"gb_yaw: ([\d.-]+)", metadata_line)
            self.current_frame_metadata["Gimbal Pitch"] = self._parse_single_metadata(
                r"gb_pitch: ([\d.-]+)", metadata_line)
            self.current_frame_metadata["Gimbal Roll"] = self._parse_single_metadata(
                r"gb_roll: ([\d.-]+)", metadata_line)

    def _save_frame_metadata(self) -> None:
        for key in DTYPES:
            if key not in self.current_frame_metadata:
                self.current_frame_metadata[key] = None
        self.extracted_data.append(self.current_frame_metadata)

    def _parse_single_metadata(self, regex: str, line: str) -> Optional[str]:
        match = re.search(regex, line)
        return match.group(1) if match else None


def get_dji_dataframe(srt_path: str) -> pd.DataFrame:
    parser = DJIMetadataParser()
    dji_data_object = parser(srt_path)
    df = pd.DataFrame(dji_data_object.metadata)
    return df


def get_valid_frame_indices(
    df: pd.DataFrame,
    delta: int = 1,
    min_pitch: float = -50,
    max_pitch: float = 90,
    min_alt: float = 0,
    threshold_gimbal: float = 0.1,
    default_focal_len: float = 24.0
) -> List[int]:
    """
    Returns a list of frame indices (sampled by `delta`) where all constraints are met.
    """
    # 1. Slice DataFrame
    df_subset = df.iloc[::delta].copy()

    # 2. Compute Masks
    mask_pitch = df_subset["Gimbal Pitch"].between(min_pitch, max_pitch)
    mask_alt = df_subset["Relative Altitude"] > min_alt
    mask_zoom = (df_subset["DZoom Ratio"] == 1.0) & (df_subset["Focal Length"] == default_focal_len)

    gimbal_cols = ["Gimbal Yaw", "Gimbal Pitch", "Gimbal Roll"]
    mask_stability = df_subset[gimbal_cols].diff().abs().le(threshold_gimbal).all(axis=1)

    # 3. Combine
    valid_mask = mask_pitch & mask_alt & mask_zoom & mask_stability

    return df_subset.index[valid_mask].tolist()


def save_valid_frames(video_path: Path, valid_indices: List[int], output_folder: Path):
    """
    Saves frames from valid_indices.
    Skips files that already exist in output_folder.
    Uses cap.grab() for high-speed skipping of unwanted/existing frames.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- 1. Filter out indices that already exist on disk ---
    needed_indices = []
    skipped_count = 0

    for idx in valid_indices:
        filename = output_folder / f"{video_path.stem}_{idx}.jpg"
        if filename.exists():
            skipped_count += 1
        else:
            needed_indices.append(idx)

    print(f"Total valid frames: {len(valid_indices)}")
    print(f"Already extracted: {skipped_count}")
    print(f"Remaining to process: {len(needed_indices)}")

    # If nothing to do, return early
    if not needed_indices:
        print("All frames already exist. Skipping video.")
        return

    # --- 2. Setup Video Capture ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Convert to set for O(1) lookup
    target_set = set(needed_indices)
    max_target = max(needed_indices)

    current_frame = 0
    saved_new_count = 0

    print(f"Processing {video_path.name}...")

    # --- 3. Optimized Processing Loop ---
    while True:
        # Optimization: Stop if we passed the last needed frame
        if current_frame > max_target:
            break

        # Check if we need this frame
        if current_frame in target_set:
            # We need this frame: Decode (expensive) and Save
            ret, frame = cap.read()
            if not ret:
                break

            filename = output_folder / f"{video_path.stem}_{current_frame}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_new_count += 1
        else:
            # We don't need this frame (or it already exists): Fast Skip
            ret = cap.grab()
            if not ret:
                break

        current_frame += 1

    cap.release()
    print(f"Done. Newly extracted: {saved_new_count}. Output: {output_folder}")


def run_pipeline(video_path: Path, config_path: str, output_folder: str) -> Optional[pd.DataFrame]:
    """
    Main orchestration function:
    1. Loads Config & SRT
    2. Calculates Valid Indices (Filtering)
    3. Retrieves Metadata for valid frames
    4. Extracts Frames (checking for duplicates)
    """
    video_path = Path(video_path)
    srt_path = video_path.parent / (video_path.stem + '.SRT')

    if not srt_path.exists():
        print(f"Error: SRT file not found at {srt_path}")
        return

    # 1. Load Config
    config = load_config(config_path)
    computer_args = config['preprocessing_args']['clips_computer_args']
    constraints = computer_args['clips_constraints']
    delta_val = computer_args.get('clip_delta', 1)

    # 2. Parse Metadata
    print("Parsing SRT metadata...")
    df = get_dji_dataframe(str(srt_path))

    # 3. Filter Frames
    print("Filtering frames based on constraints...")
    valid_indices = get_valid_frame_indices(
        df,
        delta=delta_val,
        **constraints
    )

    # 4. Retrieve Filtered Metadata
    # This gives you the dataframe containing only the "valid" rows
    df_filtered = df.loc[valid_indices].copy()

    # Optional: Save this metadata to CSV if needed
    # df_filtered.to_csv(video_path.parent / "filtered_metadata.csv", index=False)

    # 5. Extract Frames
    save_valid_frames(video_path, valid_indices, output_folder)

    return df_filtered


if __name__ == '__main__':
    # --- USAGE EXAMPLE ---
    video_path = Path('data/raw_fights/DJI_202511101344_002_Ajouter-balise5/DJI_20251110140051_0001_V.MP4')
    config_path = 'config/preprocessing/raw_preprocessing_config_refacto.yaml'

    # Run the full pipeline
    filtered_df = run_pipeline(video_path, config_path)

    if filtered_df is not None:
        print(f"\nPipeline finished. Filtered dataframe shape: {filtered_df.shape}")

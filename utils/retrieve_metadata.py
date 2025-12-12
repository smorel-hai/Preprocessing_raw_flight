import re
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as file:
        # safe_load is recommended for security
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
                    continue  # Skip keys not in DTYPES to prevent errors on extra data
                if value is None:
                    continue  # Handle None values gracefully
                try:
                    frame_data[key] = DTYPES[key](value)
                except ValueError:
                    pass  # Keep original if conversion fails


class DJIMetadataParser:
    def __call__(self, srt_path: str) -> DJIMetadata:
        return DJIMetadata(self._parse_srt(srt_path))

    def _parse_srt(self, file_path: str) -> List[Dict]:
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()

        self._split_by_frames(content)
        self.extracted_data = []

        for frame_metadata in self.frames_metadata:
            # Skip empty strings from split
            if frame_metadata.strip():
                self._parse_frame(frame_metadata)

        return self.extracted_data

    def _split_by_frames(self, content: str) -> None:
        # Split by double newline which standard SRT uses to separate blocks
        self.frames_metadata = re.split(r"\n\s*\n", content.strip())

    def _parse_frame(self, frame_metadata: str) -> None:
        self.current_frame_metadata = {}
        self._parse_frame_metadata(frame_metadata)
        if self.current_frame_metadata:  # Only save if we found data
            self._save_frame_metadata()

    def _parse_frame_metadata(self, frame_metadata: str) -> None:
        lines = frame_metadata.split("\n")

        # DJI SRTs usually have: Index, Time, FrameInfo, Timestamp, Metadata
        # We need at least 5 lines to match your regex indices safely
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
        # Ensure all keys in DTYPES exist, fill with None if missing
        for key in DTYPES:
            if key not in self.current_frame_metadata:
                self.current_frame_metadata[key] = None
        self.extracted_data.append(self.current_frame_metadata)

    def _parse_single_metadata(self, regex: str, line: str) -> Optional[str]:
        match = re.search(regex, line)
        return match.group(1) if match else None


def get_dji_dataframe(srt_path: str) -> pd.DataFrame:
    """
    Parses a DJI SRT file and returns a Pandas DataFrame.
    """
    # 1. Initialize the Parser
    parser = DJIMetadataParser()

    # 2. Parse the file (returns a DJIMetadata object)
    dji_data_object = parser(srt_path)

    # 3. Extract the list of dictionaries
    list_of_dicts = dji_data_object.metadata

    # 4. Create DataFrame
    df = pd.DataFrame(list_of_dicts)

    # Optional: Convert Timestamp string to actual datetime objects
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    return df


def get_dji_clips(
    df: pd.DataFrame,
    clip_length: int = 10,
    clip_delta: int = 60,
    min_pitch: float = -50,
    max_pitch: float = 90,
    min_alt: float = 0,
    threshold_gimbal: float = 0.1,
    default_focal_len: float = 24.0
) -> List[Tuple[int, int]]:
    """
    Returns a list of (start_index, end_index) tuples for valid clips.
    """

    # --- 1. COMPUTE MASKS (CONSTRAINTS) ---

    # A. Pitch Range
    mask_pitch = df["Gimbal Pitch"].between(min_pitch, max_pitch)

    # B. Altitude
    mask_alt = df["Relative Altitude"] > min_alt

    # C. Zoom (Digital Zoom must be 1.0 AND Focal Length must be default)
    mask_zoom = (df["DZoom Ratio"] == 1.0) & (df["Focal Length"] == default_focal_len)

    # D. Stability (Gimbal shouldn't move more than threshold between frames)
    # We check the difference between consecutive frames for Yaw, Pitch, and Roll
    gimbal_cols = ["Gimbal Yaw", "Gimbal Pitch", "Gimbal Roll"]
    # .diff() calculates change from previous row. .abs() makes it positive.
    # .le() checks if less than or equal to threshold.
    # .all(axis=1) ensures Yaw AND Pitch AND Roll are ALL stable.
    mask_stability = df[gimbal_cols].diff().abs().le(threshold_gimbal).all(axis=1)

    # --- 2. COMBINE MASKS ---

    # A frame is valid only if ALL conditions are True
    valid_mask = mask_pitch & mask_alt & mask_zoom & mask_stability

    # --- 3. EXTRACT CLIPS (COMPUTER LOGIC) ---

    # Calculate 'streaks' (cumulative count of consecutive True values)
    # This replicates the logic: if valid, streak++, else streak=0
    # We use a cumulative sum of the 'reset' points to group consecutive valid frames
    grouper = (valid_mask != valid_mask.shift()).cumsum()

    # Add a 'streak' column to the dataframe temporarily
    df['streak'] = df.groupby(grouper).cumcount() + 1
    # Where mask is False, streak should be 0 (the groupby counts everything)
    df.loc[~valid_mask, 'streak'] = 0

    # Apply the specific extraction logic from your original code:
    # "clips_ending_mask = streaks % (delta + length) == length - 1"

    # Determine where valid clips end
    # Note: We subtract 1 because cumcount starts at 0 in Python logic usually,
    # but here we simulated 1-based counting to match your loop logic.
    cycle = clip_delta + clip_length
    target_remainder = clip_length

    # Find indices where the streak hits exactly the length needed to form a clip
    # valid_ends is a boolean Series
    valid_ends = (df['streak'] >= clip_length) & (df['streak'] % cycle == target_remainder)

    # Get the integer indices
    end_indices = df.index[valid_ends].tolist()

    # Calculate start indices based on the length
    clips = [(end - clip_length + 1, end + 1) for end in end_indices]

    # Cleanup temporary column
    df.drop(columns=['streak'], inplace=True)

    return clips


def get_clips(video_path: Path, config_path: str):
    srt_path = video_path.parent / (video_path.stem + '.SRT')
    config = load_config(config_path)

    # 2. Extract the arguments neatly
    # The structure corresponds exactly to your file indentation
    computer_args = config['preprocessing_args']['clips_computer_args']
    constraints = computer_args['clips_constraints']

    df = get_dji_dataframe(srt_path)

    # 3. Call the function using "dictionary unpacking" (the ** operator)
    # This automatically maps 'clip_length' in the yaml to 'clip_length' in the function
    clips = get_dji_clips(
        df,
        clip_length=computer_args['clip_length'],
        clip_delta=computer_args['clip_delta'],
        **constraints  # This unpacks min_alt, min_pitch, etc. automatically!
    )
    print(clips)
    print(df)


if __name__ == '__main__':
    # --- USAGE EXAMPLE ---

    video_path = Path(
        '/home/sebastienmorel/Documents/Code/projects/labo/preprocessing_rawflight_pipeline/data/raw_fights/DJI_202509271440_017/DJI_20250927144136_0001_V.MP4')
    config_path = '/home/sebastienmorel/Documents/Code/projects/labo/preprocessing_rawflight_pipeline/config/preprocessing/raw_preprocessing_config_refacto.yaml'
    get_clips(video_path, config_path)

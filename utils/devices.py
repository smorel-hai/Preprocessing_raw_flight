# Copyright (c) 2025 Harmattan AI.
"""
Helper module for devices.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class IntrinsicSettings:
    """
    Intrinsic settings for a camera.
    """

    def __init__(
        self,
        K_coefs: List[float],
        distortion_coefficients: List[float],
        image_size: Tuple[int, int],
    ):
        """
        Initialize camera intrinsic settings.

        Args:
            K_coefs: Camera matrix coefficients [fx, fy, cx, cy].
            distortion_coefficients: Distortion coefficients.
            image_size: Image dimensions (width, height).
        """
        self.K_coefs = K_coefs
        self.distortion_coefficients = distortion_coefficients
        self.image_size = image_size

    def K(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get camera matrix as 3x3 numpy array.

        Args:
            shape: Shape of the image.
        """
        if shape is None:
            sx = 1
            sy = 1
        else:
            sx = shape[1] / self.image_size[1]
            sy = shape[0] / self.image_size[0]
        return np.array(
            [
                [self.K_coefs[0] * sx, 0, self.K_coefs[2] * sx],
                [0, self.K_coefs[1] * sy, self.K_coefs[3] * sy],
                [0, 0, 1],
            ]
        )

    @property
    def D(self) -> np.ndarray:
        """
        Get distortion coefficients as numpy array.

        Returns:
            np.ndarray: Distortion coefficients array.
        """
        return np.array(self.distortion_coefficients)


class Camera:
    """
    Configuration for a camera.
    """

    def __init__(
        self,
        configDict: Dict,
    ):
        """
        Initialize camera from configuration dictionary.

        Args:
            configDict: Dictionary containing camera configuration data.
        """
        data_config, intrinsic_config = self.check_config(configDict)
        self.name = configDict["camera_name"]
        self.sensor_type = configDict["sensor_type"]
        if data_config:
            self.photo_resolutions = configDict["photo_resolutions"]
            self.video_resolutions = configDict["video_resolutions"]
            self.photo_extensions = configDict["photo_extensions"]
            self.video_extensions = configDict["video_extensions"]
        if intrinsic_config:
            self.intrinsic_settings = IntrinsicSettings(
                **configDict["intrinsic_settings"]
            )
        else:
            self.intrinsic_settings = IntrinsicSettings(
                K_coefs=[1, 1, 0, 0],
                distortion_coefficients=[0, 0, 0, 0, 0],
                image_size=(1, 1),
            )

    def check_config(self, configDict: Dict):
        """
        Validate camera configuration dictionary.

        Args:
            configDict: Dictionary containing camera configuration data.

        Returns:
            Tuple[bool, bool]: (data_config_valid, intrinsic_config_valid).
        """
        data_config = True
        intrinsic_config = True
        if "photo_resolutions" not in configDict:
            data_config = False
        if "video_resolutions" not in configDict:
            data_config = False
        if "photo_extensions" not in configDict:
            data_config = False
        if "video_extensions" not in configDict:
            data_config = False
        if "intrinsic_settings" not in configDict:
            intrinsic_config = False
        if not data_config:
            logger.warning(
                """Invalid resolution and extension settings for camera:
                %s. Skipping camera.""",
                configDict["camera_name"],
            )
        if not intrinsic_config:
            logger.warning(
                """Invalid intrinsic settings for camera:
                %s. Setting default intrinsic settings.""",
                configDict["camera_name"],
            )
        return data_config, intrinsic_config


class Device:
    """
    Configuration for a device.
    """

    def __init__(
        self,
        serial_number: str,
        cameras: Dict[str, Camera],
        homographies: Dict[str, List[List[float]]],
    ):
        """
        Initialize device with serial number, cameras, and homographies.

        Args:
            serial_number: Device serial number.
            cameras: Dictionary of camera configurations.
            homographies: Dictionary of homography matrices between cameras.
        """
        self.serial_number = serial_number
        self.cameras = cameras
        self.homographies = homographies

    def add_camera(self, camera: Camera):
        """
        Add a camera to the device.

        Args:
            camera: Camera object to add.
        """
        self.cameras.append(camera)

    def modify_camera(self, camera: Camera):
        """
        Modify an existing camera configuration.

        Args:
            camera: Camera object with updated configuration.
        """
        for i, c in enumerate(self.cameras):
            if c.name == camera.name:
                self.cameras[i] = camera
                break

    def remove_camera(self, camera: Camera):
        """
        Remove a camera from the device.

        Args:
            camera: Camera object to remove.
        """
        self.cameras = [c for c in self.cameras if c.name != camera.name]

    def add_homography(self, homography: str, camera_pair: Tuple[str, str]):
        """
        Add homography matrix for camera pair.

        Args:
            homography: Homography matrix data.
            camera_pair: Tuple of (source_camera, dest_camera) names.
        """
        key = f"{camera_pair[0]}_{camera_pair[1]}"
        self.homographies[key] = homography

    def modify_homography(self, homography: str, camera_pair: Tuple[str, str]):
        """
        Modify homography matrix for camera pair.

        Args:
            homography: Updated homography matrix data.
            camera_pair: Tuple of (source_camera, dest_camera) names.
        """
        key = f"{camera_pair[0]}_{camera_pair[1]}"
        self.homographies[key] = homography

    def remove_homography(self, camera_pair: Tuple[str, str]):
        """
        Remove homography matrix for camera pair.

        Args:
            camera_pair: Tuple of (source_camera, dest_camera) names.
        """
        key = f"{camera_pair[0]}_{camera_pair[1]}"
        self.homographies.pop(key)


def load_device_config(config_path: str, serial_number: str) -> Device:
    """
    Load device configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.
        serial_number: Device serial number to load.

    Returns:
        Device: Configured device object.

    Raises:
        ValueError: If serial number not found in config file.
    """
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if serial_number in config_data:
        data = config_data[serial_number]
    else:
        raise ValueError(
            """Serial number %s not found in config file %s""",
            serial_number,
            config_path,
        )

    cameras = {}
    for camera_serial_number in data["cameras"]:
        camera = data["cameras"][camera_serial_number]
        cameras[camera_serial_number] = Camera(configDict=camera)

    if "homographies" in data:
        if isinstance(data["homographies"], dict):
            homographies = {
                key: np.array(value)
                for key, value in data["homographies"].items()
            }
        else:
            logger.warning(
                """Invalid homographies format for device:
                %s. Skipping homographies.""",
                serial_number,
            )
            homographies = None
    else:
        homographies = None

    return Device(
        serial_number=serial_number, cameras=cameras, homographies=homographies
    )


def save_device_config(device: Device, config_path: str):
    """
    Save device configuration to YAML file.

    Args:
        device: Device object to save.
        config_path: Path to save the YAML configuration file.
    """
    with open(config_path, "w") as f:
        yaml.dump(device.__dict__, f, default_flow_style=False, indent=2)

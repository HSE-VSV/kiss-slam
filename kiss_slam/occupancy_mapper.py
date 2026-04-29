# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import open3d as o3d
import yaml
from kiss_icp.voxelization import voxel_down_sample
from PIL import Image

from kiss_slam.config import OccupancyMapperConfig
from kiss_slam.kiss_slam_pybind import kiss_slam_pybind


class OccupancyGridMapper:
    def __init__(
        self,
        config: OccupancyMapperConfig,
    ):
        self.config = config
        self.occupancy_mapping_pipeline = kiss_slam_pybind._OccupancyMapper(self.config.resolution)
        self.has_intensity = False
        self._intensity_sum_by_voxel = defaultdict(float)
        self._intensity_count_by_voxel = defaultdict(int)

    def integrate_frame(
        self, frame: np.ndarray, pose: np.ndarray, intensities: Optional[np.ndarray] = None
    ):
        frame_downsampled = voxel_down_sample(frame, self.config.resolution).astype(np.float32)
        self.occupancy_mapping_pipeline._integrate_frame(
            kiss_slam_pybind._Vector3fVector(frame_downsampled), pose
        )
        if intensities is not None:
            self._integrate_intensities(frame, pose, intensities)

    def _integrate_intensities(
        self, frame: np.ndarray, pose: np.ndarray, intensities: np.ndarray
    ):
        if len(frame) != len(intensities):
            raise ValueError(
                f"Intensity length mismatch: got {len(intensities)} intensities "
                f"for {len(frame)} points"
            )

        self.has_intensity = True
        intensities = np.asarray(intensities)
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        world_points = frame @ rotation.T + translation
        valid = np.isfinite(world_points).all(axis=1) & np.isfinite(intensities)
        voxel_indices = np.floor(world_points[valid] / self.config.resolution).astype(np.int32)
        valid_intensities = intensities[valid].astype(np.float64)

        for voxel, intensity in zip(voxel_indices, valid_intensities):
            key = tuple(int(index) for index in voxel)
            self._intensity_sum_by_voxel[key] += float(intensity)
            self._intensity_count_by_voxel[key] += 1

    def compute_3d_occupancy_information(self):
        active_voxels, occupancies = self.occupancy_mapping_pipeline._get_active_voxels()
        self.active_voxels = np.asarray(active_voxels, np.int32)
        self.occupancies = np.asarray(occupancies, float)
        self.occupied_voxels = self.active_voxels[
            np.where(self.occupancies > self.config.occupied_threshold)[0]
        ]

    def compute_3d_occupied_voxels(self):
        occupied_voxels = self.occupancy_mapping_pipeline._get_occupied_voxels(
            self.config.occupied_threshold
        )
        self.occupied_voxels = np.asarray(occupied_voxels, np.int32)

    def compute_2d_occupancy_information(self):
        min_z_idx = int(self.config.z_min // self.config.resolution)
        max_z_idx = int(self.config.z_max // self.config.resolution)

        indices_in_range = np.where(
            (self.active_voxels[:, 2] <= max_z_idx) & (self.active_voxels[:, 2] >= min_z_idx)
        )[0]
        voxels_in_range = self.active_voxels[indices_in_range]

        self.lower_bound = np.min(voxels_in_range, 0)
        self.upper_bound = np.max(voxels_in_range, 0)
        nrows, ncols, nslices = self.upper_bound - self.lower_bound + 1
        occupancy_grid = np.ones((nrows, ncols, nslices)) * 0.5

        occupancy_grid[
            voxels_in_range[:, 0] - self.lower_bound[0],
            voxels_in_range[:, 1] - self.lower_bound[1],
            voxels_in_range[:, 2] - self.lower_bound[2],
        ] = self.occupancies[indices_in_range]
        self.occupancy_grid = 1.0 - np.max(occupancy_grid, 2)

    def write_3d_occupancy_ply(self, output_dir):
        map_points = (0.5 + self.occupied_voxels) * self.config.resolution
        o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(map_points))
        o3d_pcd.estimate_normals()
        output_path = os.path.join(output_dir, "occupancy_pcd.ply")
        if not self.has_intensity:
            o3d.io.write_point_cloud(output_path, o3d_pcd)
            return
        self._write_occupancy_ply_with_intensity(
            output_path, map_points, np.asarray(o3d_pcd.normals)
        )

    def _occupied_voxel_intensities(self):
        values = np.zeros(len(self.occupied_voxels), dtype=np.float32)
        missing = 0
        for idx, voxel in enumerate(self.occupied_voxels):
            key = tuple(int(index) for index in voxel)
            count = self._intensity_count_by_voxel.get(key, 0)
            if count > 0:
                values[idx] = self._intensity_sum_by_voxel[key] / count
            else:
                missing += 1

        if missing:
            print(
                f"KissSLAM| Warning: {missing} occupied voxels had no intensity observations; "
                "wrote intensity=0.0"
            )
        return values

    def _write_occupancy_ply_with_intensity(
        self, output_path: str, map_points: np.ndarray, normals: np.ndarray
    ):
        intensities = self._occupied_voxel_intensities()
        with open(output_path, "w") as ply_file:
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {len(map_points)}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("property float nx\n")
            ply_file.write("property float ny\n")
            ply_file.write("property float nz\n")
            ply_file.write("property float intensity\n")
            ply_file.write("end_header\n")
            for point, normal, intensity in zip(map_points, normals, intensities):
                ply_file.write(
                    f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                    f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} "
                    f"{float(intensity):.9g}\n"
                )

    def write_3d_occupancy_boxai_volume(self, output_dir):
        self.occupancy_mapping_pipeline._save_occupancy_volume(
            os.path.join(output_dir, "occupancy_grid_bonxai.bin")
        )

    def write_2d_occupancy_grid(self, output_dir):
        image_filename = "map.png"
        occupancy_image = np.rot90(self.occupancy_grid, 0) * 255.0
        Image.fromarray(np.asarray(occupancy_image, np.uint8)).save(
            os.path.join(output_dir, image_filename)
        )
        grid_info = {
            "image": image_filename,
            "resolution": self.config.resolution,
            "origin": [
                float(self.lower_bound[0]) * self.config.resolution,
                float(self.lower_bound[1]) * self.config.resolution,
                0.0,
            ],
            "occupied_thresh": self.config.occupied_threshold,
            "free_thresh": self.config.free_threshold,
            "negate": 0,
        }
        with open(os.path.join(output_dir, "map.yaml"), "w") as yaml_file:
            yaml.dump(grid_info, yaml_file, default_flow_style=False, sort_keys=False)

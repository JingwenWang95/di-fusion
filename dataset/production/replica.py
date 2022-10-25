import cv2
import os
import glob
import torch
from dataset.production import *
from pyquaternion import Quaternion
from pathlib import Path
from utils import motion_util


class ReplicaSequence(RGBDSequence):
    def __init__(self, path: str, start_frame: int = 0, end_frame: int = -1):
        super().__init__()
        self.path = Path(path)
        self.color_names = sorted(glob.glob(f'{self.path}/results/frame*.jpg'))
        self.depth_names = sorted(glob.glob(f'{self.path}/results/depth*.png'))
        # fx, fy, cx, cy, depth_scale
        self.calib = [600.0, 600.0, 599.5, 339.5, 6553.5]

        self._parse_traj_file("{}/traj.txt".format(self.path))
        self.first_iso = self.poses[0]

        if end_frame == -1:
            end_frame = len(self.color_names)

        self.color_names = self.color_names[start_frame:end_frame]
        self.depth_names = self.depth_names[start_frame:end_frame]

        # if load_gt:
        #     gt_traj_path = (list(self.path.glob("*.freiburg")) + list(self.path.glob("groundtruth.txt")))[0]
        #     self.gt_trajectory = self._parse_traj_file(gt_traj_path)
        #     self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
        #     change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())
        #     self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
        #     assert len(self.gt_trajectory) == len(self.color_names)
        # else:
        #     self.gt_trajectory = None

    def _parse_traj_file(self, traj_path):
        self.poses = []
        with open(traj_path, "r") as f:
            lines = f.readlines()
        for i in range(len(self.color_names)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            pose_iso = motion_util.Isometry.from_matrix(c2w)
            self.poses.append(pose_iso)

    def __len__(self):
        return len(self.color_names)

    def __next__(self):
        if self.frame_id >= len(self):
            raise StopIteration

        depth_img_path = self.path / self.depth_names[self.frame_id]
        rgb_img_path = self.path / self.color_names[self.frame_id]

        # Convert depth image into point cloud.
        depth_data = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() / self.calib[4]
        rgb_data = cv2.imread(str(rgb_img_path))
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.

        frame_data = FrameData()
        frame_data.gt_pose = self.poses[self.frame_id]
        frame_data.calib = FrameIntrinsic(self.calib[0], self.calib[1], self.calib[2], self.calib[3], self.calib[4])
        frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        self.frame_id += 1
        return frame_data

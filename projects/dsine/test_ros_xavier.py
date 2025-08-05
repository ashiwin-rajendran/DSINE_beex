#!/usr/bin/env python3

import os
import sys
import numpy as np
import glob

import torch
import torch.nn.functional as F

from torchvision import transforms
import cv2

# Add the project root to Python path and change working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# Store original paths
original_cwd = os.getcwd()
original_sys_path = sys.path.copy()

# Set up paths for imports
sys.path.insert(0, project_root)
os.chdir(project_root)

try:
    # Now import project modules
    import utils.utils as utils
    import utils.visualize as vis_utils

    # NOTE: project-specific imports (e.g. config)
    import projects.dsine.config as config

    # Skip dataloader import as it's not needed for ROS mode
    # from projects.baseline_normal.dataloader import *

    from utils.projection import intrins_from_fov, intrins_from_txt

finally:
    # Restore original working directory
    os.chdir(original_cwd)

import rospy
from sensor_msgs.msg import CompressedImage
import threading


class ROSInputStream:
    def __init__(self, input_topic, output_topic, device):
        self.device = device
        self.latest_image = None
        self.image_lock = threading.Lock()

        # ROS setup
        rospy.init_node("surface_normal_estimation", anonymous=True)
        self.subscriber = rospy.Subscriber(input_topic, CompressedImage, self.image_callback)
        self.publisher = rospy.Publisher(output_topic, CompressedImage, queue_size=1)

        # Image processing setup
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def image_callback(self, msg):
        try:
            # Decode compressed image directly without cv_bridge
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.resize(cv_image, (640, 360))
            with self.image_lock:
                self.latest_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def get_sample(self):
        with self.image_lock:
            if self.latest_image is None:
                return None
            color_image = self.latest_image.copy()

        # Convert to RGB and normalize
        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Padding
        _, _, orig_H, orig_W = img.shape
        lrtb = utils.get_padding(orig_H, orig_W)
        img = F.pad(img, lrtb, mode="constant", value=0.0)
        img = self.normalize(img)

        # Intrinsics (assuming 60 degree FOV)
        intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
        intrins[:, 0, 2] += lrtb[0]
        intrins[:, 1, 2] += lrtb[2]

        self.lrtb = lrtb
        self.new_H, self.new_W = orig_H, orig_W

        return {"color_image": color_image, "img": img, "intrins": intrins}

    def publish_result(self, normal_rgb):
        try:
            # Convert normal RGB to BGR for OpenCV
            normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)

            # Create compressed image message
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"

            # Encode image
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, encimg = cv2.imencode(".jpg", normal_bgr, encode_param)
            msg.data = encimg.tobytes()

            # Publish
            self.publisher.publish(msg)
        except Exception as e:
            rospy.logerr(f"Error publishing result: {e}")


if __name__ == "__main__":
    device = torch.device("cuda")

    # Get args before changing directory
    args = config.get_args(test=True)

    # Convert relative checkpoint path to absolute path before changing directory
    if args.ckpt_path and not os.path.isabs(args.ckpt_path):
        args.ckpt_path = os.path.abspath(args.ckpt_path)

    # Change to project root for model loading
    os.chdir(project_root)

    print(f"Args output_dir: {args.output_dir}")
    print(f"Args ckpt_path: {args.ckpt_path}")

    if args.ckpt_path is None:
        model_dir = os.path.join(args.output_dir, "models")
        print(f"Looking for models in: {model_dir}")
        ckpt_paths = glob.glob(os.path.join(model_dir, "*.pt"))
        print(f"Found checkpoint files: {ckpt_paths}")
        if ckpt_paths:
            ckpt_paths.sort()
            args.ckpt_path = ckpt_paths[-1]
            print(f"Selected checkpoint: {args.ckpt_path}")
        else:
            print("No .pt files found in models directory")

    print(f"Final checkpoint path: {args.ckpt_path}")
    print(f"Checkpoint exists: {os.path.exists(args.ckpt_path) if args.ckpt_path else False}")

    if not args.ckpt_path or not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")

    # assert os.path.exists(args.ckpt_path)

    # NOTE: define and load model
    if args.NNET_architecture == "v00":
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == "v01":
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == "v02":
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == "v02_kappa":
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception("invalid arch")
    model = DSINE(args).to(device)

    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()

    if args.mode == "ros":
        if not hasattr(args, "input_topic") or not hasattr(args, "output_topic"):
            raise Exception("ROS mode requires --input_topic and --output_topic arguments")

        InputStream = ROSInputStream(args.input_topic, args.output_topic, device)

        rospy.loginfo(f"Starting ROS surface normal estimation")
        rospy.loginfo(f"Input topic: {args.input_topic}")
        rospy.loginfo(f"Output topic: {args.output_topic}")

        rate = rospy.Rate(30)  # 30 Hz

        while not rospy.is_shutdown():
            with torch.no_grad():
                data_dict = InputStream.get_sample()
                if data_dict is None:
                    rate.sleep()
                    continue

                # Forward pass (same as demo function)
                img = data_dict["img"]
                intrins = data_dict["intrins"]

                norm_out = model(img, intrins=intrins, mode="test")[-1]
                norm_out = norm_out[
                    :,
                    :,
                    InputStream.lrtb[2] : InputStream.lrtb[2] + InputStream.new_H,
                    InputStream.lrtb[0] : InputStream.lrtb[0] + InputStream.new_W,
                ]
                pred_norm = norm_out[:, :3, :, :]

                # Convert to RGB for publishing
                pred_norm_rgb = vis_utils.normal_to_rgb(pred_norm)[0, ...]

                # Publish result
                InputStream.publish_result(pred_norm_rgb)

            rate.sleep()
        exit()

    else:
        raise Exception("invalid input option for demo")

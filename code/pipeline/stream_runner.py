import time
import logging
import cv2
import torch
import numpy as np
import redis
import json
import os

from tracker.deep_sort import deepsort_update
from utils.draw_utils import draw_bbox_skeleton

logger = logging.getLogger(__name__)

class StreamRunner:
    def __init__(self, config, detector, tracker, keypoint_estimator):
        self.detector = detector
        self.tracker = tracker
        self.keypoint_estimator = keypoint_estimator
        self.detector_config = config.get("detector", {})
        stream_cfg = config.get("stream", {})
        self.video_path = stream_cfg.get("source")
        self.output_mode = stream_cfg.get("output_mode")

        # VIDEO 설정
        self.save_video = (self.output_mode == "video")
        if self.save_video:
            video_cfg = stream_cfg.get("video", {})
            self.output_path = video_cfg.get("output_path", "output.mp4")
            self.fps = video_cfg.get("fps", 30)
            self.writer = None
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        # REDIS 설정
        self.send_redis = (self.output_mode == "redis")
        if self.send_redis:
            redis_cfg = stream_cfg.get("redis", {})
            self.redis_client = redis.StrictRedis(
                host=redis_cfg.get("host", "127.0.0.1"),
                port=redis_cfg.get("port", 6379),
                db=0
            )
        logger.info(f"Output mode configured → "
            f"(selected: {self.output_mode})")

    def run(self):
        self.cap = self.open_stream()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.info("End of video stream or cannot read frame. Exiting...")
                break
            
            bbox_keypoints_info = self.model_pipeline(frame)

            if self.save_video:
                frame_drawn = draw_bbox_skeleton(frame, bbox_keypoints_info)
                self.save_frame(frame_drawn)

            if self.send_redis:
                serialized = self.serialize_for_redis(bbox_keypoints_info)
                self.redis_client.publish("keypoints_result", serialized)

        if self.save_video and self.writer is not None:
            self.writer.release()
            logger.info(f"Video saved to: {self.output_path}")

    def open_stream(self):
        logger.info(f"Opening video source: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.video_path}. "
                            "Check the path.")
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        logger.info(f"Successfully opened video source: {self.video_path}")
        return cap
    
    def model_pipeline(self, frame):
        results = self.detector.predict(frame,
                                        verbose=False,
                                        conf=self.detector_config["confidence"],
                                        iou=self.detector_config["iou_threshold"],
                                        max_det=self.detector_config["max_det"],
                                        device=self.detector_config["device"],
                                        classes=[0])

        detections = results[0].boxes

        if detections is None or detections.shape[0] == 0:
            return {}

        xyxy = detections.xyxy
        xywh = detections.xywh
        conf = detections.conf.unsqueeze(-1)
        cls  = detections.cls.unsqueeze(-1)

        # tracking input
        track_boxes = torch.cat((xyxy, conf, cls), dim=1)
        pose_boxes  = torch.cat((xywh, conf, cls), dim=1)

        deepsort_pred = deepsort_update(
            self.tracker,
            track_boxes.cpu(),
            pose_boxes[:, 0:4].cpu(),
            frame
        )

        if deepsort_pred is None or len(deepsort_pred) == 0:
            return {}

        detection_pred = deepsort_pred.astype(np.float32)

        bbox_keypoints_info = self.keypoint_estimator.keypoint_estimate(frame, detection_pred)

        return bbox_keypoints_info

    def serialize_for_redis(self, bbox_keypoints_info):
        serializable = {}
        for obj_id, info in bbox_keypoints_info.items():
            keypoints, confs = info["keypoints"]

            serializable[obj_id] = {
                "boxes": list(map(float, info["boxes"])),
                "keypoints": keypoints.tolist(),
                "confidence": confs.tolist()
            }

        return json.dumps(serializable)

    def save_frame(self, frame):
        if self.writer is None:
            h, w, _ = frame.shape
            self.writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (w, h)
            )
        self.writer.write(frame)
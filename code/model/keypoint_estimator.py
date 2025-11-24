import logging

import cv2
import numpy as np
from typing import Tuple

from model.trt_engine import TRTEngine

logger = logging.getLogger(__name__)


class KeypointEstimator:
    def __init__(self, weight):
        logger.info(f"Keypoint Estimater Initialize")
        self.pose_model_path = weight

        # Pose Model Load
        try:
            self.pose_model = TRTEngine(self.pose_model_path)
            logger.info(f"Succes to Load Pose Estimation Model")
        except Exception as e:
            logger.error(f"Fail to Load Pose Estimation Model: {e}", exc_info=True)

    def keypoint_estimate(self, raw_image, tracked_objects):
        keypoint_results_per_frame = {}

        for tracked_obj in tracked_objects:
            object_id = int(tracked_obj[5])
            xmin, ymin, xmax, ymax = [int(coord) for coord in tracked_obj[:4]]

            person_img = raw_image[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

            origin_h, origin_w = person_img.shape[:2]

            image = self.pre_processing(person_img)

            keypoints_results = self.keypoints_estimator_forward(self.pose_model(image), 
                                                                 origin_h, origin_w, [xmin, ymin, xmax, ymax])
            
            keypoint_results_per_frame[object_id] = {'keypoints': keypoints_results,'boxes':tracked_obj[:4]}

        return keypoint_results_per_frame

    def pre_processing(self, person_img):
        image = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
        resized_image = cv2.resize(person_img, (192, 256))  # 모델의 입력 크기에 맞게 이미지 크기 조정
        normalized_image = resized_image.astype(np.float32) / 255.0  # 이미지 정규화
        input_data = normalized_image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        input_data = np.expand_dims(input_data, axis=0)  # (C, H ,W) -> (1, C ,H ,W)
        image = np.ascontiguousarray(input_data) 
        return image
    
    def keypoints_estimator_forward(self, pose_result, origin_height, origin_width, coord):
        x1, y1, x2, y2 = coord

        simcc_x=pose_result[0].reshape((1, 17, 384))
        simcc_y=pose_result[1].reshape((1, 17, 512))

        keypoints, score = self.get_simcc_maximum(simcc_x,simcc_y)
        keypoint_labels = keypoints

        keypoints_17 = keypoint_labels[0]*np.array([origin_width/384,origin_height/512])
        keypoints_17=self.convert_keypoints_to_full_frame([x1,y1,x2,y2],keypoints_17)

        return (keypoints_17, score)
    
    def convert_keypoints_to_full_frame(self, cropped_box, keypoints_cropped):
        x1_cropped, y1_cropped, _, _ = cropped_box
        x_offset = x1_cropped
        y_offset = y1_cropped

        offset_matrix = np.array([[x_offset, y_offset]])
        keypoints_original = keypoints_cropped + offset_matrix

        return keypoints_original
    
    def get_simcc_maximum(self, simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get maximum response location and value from simcc representations.

        Note:
            instance number: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
            simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

        Returns:
            tuple:
            - locs (np.ndarray): locations of maximum heatmap responses in shape
                (K, 2) or (N, K, 2)
            - vals (np.ndarray): values of maximum heatmap responses in shape
                (K,) or (N, K)
        """

        assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
        assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')

        assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
            f'Invalid shape {simcc_x.shape}')
        
        assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
            f'Invalid shape {simcc_y.shape}')
        
        assert simcc_x.ndim == simcc_y.ndim, (
            f'{simcc_x.shape} != {simcc_y.shape}')

        if simcc_x.ndim == 3:
            N, K, Wx = simcc_x.shape
            simcc_x = simcc_x.reshape(N * K, -1)
            simcc_y = simcc_y.reshape(N * K, -1)
        else:
            N = None

        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)

        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        vals = max_val_x
        locs[vals <= 0.] = -1

        if N:
            locs = locs.reshape(N, K, 2)
            vals = vals.reshape(N, K)

        return locs, vals
    
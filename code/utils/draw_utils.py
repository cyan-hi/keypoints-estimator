import cv2
import numpy as np
import os

# COCO-17 keypoint skeleton 연결 정보
COCO17_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # 얼굴/팔
    (0, 5), (0, 6), (5, 7), (7, 9),      # 왼쪽 몸통-팔
    (6, 8), (8, 10),                      # 오른쪽 몸통-팔
    (5, 6), (5, 11), (6, 12),             # 몸통-골반
    (11, 12), (11, 13), (12, 14),         # 다리
    (13, 15), (14, 16)                    # 발
]
def draw_bbox_skeleton(frame, bbox_keypoints_info):
    """
    frame: BGR 이미지
    bbox_keypoints_info: dict {id: {'keypoints': (coords, conf), 'boxes': [x1,y1,x2,y2]}}
    """
    for obj_id, info in bbox_keypoints_info.items():
        keypoints, confs = info['keypoints']
        x1, y1, x2, y2 = info['boxes']

        keypoints = np.array(keypoints)
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        elif keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 2)

        confs = np.array(confs)
        if confs.ndim == 2:
            confs = confs[0]

        # bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f'ID:{obj_id}', (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # keypoints
        for i, (x, y) in enumerate(keypoints):
            if confs[i] > 0.1:
                cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), -1)

        # skeleton
        for start_idx, end_idx in COCO17_CONNECTIONS:
            if confs[start_idx] > 0.1 and confs[end_idx] > 0.1:
                pt1 = tuple(keypoints[start_idx].astype(int))
                pt2 = tuple(keypoints[end_idx].astype(int))
                cv2.line(frame, pt1, pt2, (0,255,0), 2)

    return frame


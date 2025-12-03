# Real-Time Multi-Person Tracking & Keypoints Estimation Pipeline
**YOLOv11 + DeepSORT + RTMPose-m (TensorRT)**  
**GTX 1650 Super(4GB) 기준 최대 20.8 FPS 실시간 처리**
---
<p align="center"> <a href="output.gif"> <img src="output.gif" width="650" /> </a> </p>

## Overview
본 프로젝트는 **사람 검지 → 추적 → 17 Keypoints 추정 → 영상 출력 또는 Redis 송신**까지  
실시간으로 처리하는 End-to-End 파이프라인입니다.

- YOLOv11: 사람 검지  
- DeepSORT: Tracking ID 유지  
- RTMPose-m: 17 Keypoints 추론  
- TensorRT 기반 속도 최적화  
- Docker + Redis 연동

**GTX 1650 Super(4GB)** 환경에서 **20FPS 이상 성능 확보**를 목표로 최적화했습니다.

---

## Performance (GTX 1650 SUPER)

| Detector | Pose Engine | FPS | Notes |
|---------|-------------|-----|-------|
| YOLO11m (PyTorch) | rtmpose_m.engine | 16.5 | PyTorch baseline |
| yolov11m_fp32.engine | rtmpose_m.engine | 18.2 | TensorRT FP32 |
| yolov11m_fp16.engine | rtmpose_m.engine | 19.1 | Detector FP16 |
| **yolov11m_fp16.engine** | **rtmpose_m_fp16.engine** | **20.8** | **최종 최적화** |

---

## Pipeline Architecture

```
[ Video Input ]
      ↓
[ YOLO11 Detector ]        # PyTorch or TensorRT engine
      ↓
(bbox, score)
      ↓
[ DeepSORT Tracker ]       # track_id
      ↓
[ RTMPose-m Keypoints ]    # TensorRT engine (17 joints)
      ↓
(keypoints)
      ↓
[ Visualization / Redis Publish ]
```

---

## Technical Highlights

### ✔ TensorRT 엔진 최적화 (FP32 → FP16)
- YOLOv11 / RTMPose 모두 엔진 변환  
- 약 12~20% 성능 향상  
- 메모리 사용 최적화 (약 2.2~2.8GB)

### ✔ Keypoints 후처리 로직 직접 구현
- heatmap → 좌표 변환  
- bbox alignment  
- skeleton 연결  
- 시각화 렌더링

### ✔ 실시간 파이프라인 설계
- frame 단위 처리 흐름  
- non-blocking  
- DeepSORT ReID distance 튜닝

### ✔ Redis Publish 지원
- track_id + keypoints + bbox를 JSON 형태로 송신  
- 실시간 제어 시스템과 연동 가능

### ✔ Docker 기반 실행환경 제공
- python-app + redis 멀티 컨테이너  
- reproducible environment

---

## ▶ 실행 방법

### 1) Clone
```
git clone https://github.com/cyan-hi/keypoints-estimator.git
cd keypoints-estimator
```

### 2) Docker 실행
```
docker-compose up --build -d
docker-compose exec python-app bash
```

### 3) TensorRT 엔진 생성 (최초 1회)
```
python weights/make_engines.py
```

### 4) 실행
```
python main.py
```

---

## Config Example

```json
{
    "stream": {
        "source": "/app/video/test.mp4",
        "output_mode": "redis",
        "video": { 
            "output_path": "/app/output/output.mp4",
            "fps": 30
        },
        "redis": { 
            "host": "redis",
            "port": 6379,
            "channel": "ai-result"
        }
    },
    "detector": {
        "weight": "/app/weights/yolov11_m_fp16.engine",
        "confidence": 0.35,
        "iou_threshold": 0.75
    },
    "keypoint_estimator": {
        "weight": "/app/weights/rtmpose_m_fp16.engine"
    }
}
```

---

## Directory Structure

```
keypoints-estimator/
├── code
│   ├── cfg/
│   ├── utils/
│   ├── model/
│   ├── pipeline/
│   ├── tracker/
│   ├── weights/
│   └── main.py
├── docker/
├── docker-compose.yml
└── README.md
```

---

## Requirements

```
shapely==2.0.6
scipy==1.14.1
pandas==2.2.3
numpy==1.23.5
requests==2.32.3
tqdm==4.67.0
seaborn==0.13.2
matplotlib==3.9.2
PyYAML==6.0.2
opencv-python==4.10.0.84
psutil==6.1.0
protobuf==4.21.2
ipython==8.29.0
ultralytics==8.3.57
redis==7.1.0
onnxscript==0.5.6
```

---

## References

- YOLO (Ultralytics)  
  https://github.com/ultralytics/ultralytics  
- RTMPose (MMPose)  
  https://github.com/open-mmlab/mmpose  
- DeepSORT  
  https://github.com/nwojke/deep_sort  
- Test Video: AI-Hub  


# 실시간 사람 추적 & 17점 키포인트 추정 시스템  
**YOLOv11 + DeepSORT + RTMPose-m (TensorRT)**  
**GTX 1650 Super (4GB) 실측 20.8 FPS**

## 한 줄 요약
GTX 1650 Super 한 대로 **YOLOv11 + DeepSORT + RTMPose 풀 파이프라인 실시간 구동 성공**  

## GTX 1650 Super 실측 성능 (전체 파이프라인 포함)
| Detector 엔진                | Pose 엔진                   | 실측 FPS     | 비고                          |
|------------------------------|-----------------------------|--------------|-------------------------------|
| yolov11m.pt (PyTorch)        | rtmpose_m.engine            | 16.5 FPS     | PyTorch 기준                  |
| yolov11m_fp32.engine         | rtmpose_m.engine            | 18.2 FPS     | TensorRT FP32                 |
| yolov11m_fp16.engine         | rtmpose_m.engine            | 19.1 FPS     | Detector만 FP16               |
| **yolov11m_fp16.engine**     | **rtmpose_m_fp16.engine**   | **20.8 FPS** | **최적화 완료 → 실시간 돌파!** |

→ **1650 Super 4GB로도 20.8 FPS**
## 실행 흐름도
```scss
[ Video / Stream ]
        ↓
[ YOLO11 Detector ] #pt, engine
        ↓ (bbox, score)
[ DeepSORT Tracker ]
        ↓ (track_id)
[ RTMPose Keypoint Estimator ] #engine
        ↓ (17 keypoints)
[ Redis Publish ]
```
## 실행 방법
```bash
# 1. 클론 받기
git clone https://github.com/yourname/your-repo.git
cd your-repo

# 2. 처음 한 번만 (Docker 이미지 빌드 + 컨테이너 시작)
docker-compose up --build -d
# (시간이 좀 걸릴 수 있음)

# 3. 컨테이너 안으로 들어가기
docker-compose exec app bash

# 4. 실행 (진짜 이 한 줄!)
# config에서 실행 할 영상 파일 및 모델 파일 경로 확인
# config에서 output 형태 확인 ( "video" or "redis")
python main.py
```

## Config 예시 및 설명
```json
{
    "stream": {
        "source": "/app/video/test.mp4", // **영상 경로
        "output_mode": "redis", // video(video 저장) or redis(redis 전송)
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
        "weight": "/app/weights/yolov11_m_fp16.engine", //yolo 모델 파일 경로
        "device": "cuda:0",
        "confidence": 0.35,
        "iou_threshold": 0.75,
        "max_det": 10
    },
    "keypoint_estimator": {
        "weight": "/app/weights/rtmpose_m_fp16.engine" //rtmpose 모델 파일 경로
    },
    "tracker": {
        "reid_checkpoint": "/app/tracker/deep/checkpoint/ckpt.t7",
        "max_cosine_distance": 0.6,
        "min_confidence": 0.3,
        "nms_max_overlap": 0.2,
        "max_iou_distance": 0.7,
        "max_age": 70,
        "n_init": 3,
        "nn_budget": null,
        "use_appearence": true,
        "use_cuda": true
    }
}
```

## 프로젝트 구조
```plaintext
keypoints-estimator/
├── code
│   ├── cfg/                      # config 파일
│   ├── utils/                    # 유틸리티 코드
│   │   ├── config_parser.py      # 설정 파일 파싱
│   │   └── logger.py             # 로깅 유틸리티
│   ├── model/                    # ai 모델 관련
│   │   ├── keypoint_estimator.py # keypoint estimator
│   │   └── trt_engine.py         # tensorRT 엔진 로더
│   ├── pipeline/                 
│   │   └── stream_runner.py      # 모델 추론 및 파이프라인 관리
│   ├── tracker/                  # deepsort 관련 코드
│   └── main.py                   # 메인 실행 파일
├── docker                        # docker 폴더
│   ├── python-app                # python-app 도커파일
│   │   └── dockerfile
│   └── redis
│       ├── dockerfile            # redis 도커파일
│       └── redis.conf            # redis 설정 파일
├── docker-compose.yml            # 도커컴포즈 파일
└── README.md                     # 프로젝트 설명
``` 

## Requirements
```plaintext
shapely==2.0.6
scipy==1.14.1
pandas==2.2.3
numpy==1.23.5
requests==2.32.3
tqdm==4.67.0
seaborn==0.13.2
matplotlib==3.9.2
PyYAML==6.0.2
opencv-python == 4.10.0.84
psutil==6.1.0
protobuf==4.21.2
ipython==8.29.0
ultralytics==8.3.57
redis==7.1.0
```

## Docker 없이 로컬에서 돌리고 싶으신 분들
로컬 환경은 CUDA 11.8~12.x + TensorRT 8.6 + 정확한 의존성 버전이 필요해서
테스트 환경마다 다르게 동작할 수 있습니다.

## 필요시 
- trtexec --onnx=model.onnx --saveEngine=model_fp16.engine



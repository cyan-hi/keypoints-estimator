# Portfolio – 실시간 사람 추적 & 키포인트 추정 시스템

**YOLO11 + DeepSORT + RTMPose(TensorRT)** 추론 파이프라인

---

## 한 줄 요약
파일 입력 → 사람 탐지 → DeepSORT 추적 → 17점 키포인트 실시간 추정까지 한 번에 처리하는 엔드투엔드 시스템 (Docker 완벽 지원)

---

## 주요 특징
- YOLO11m (Ultralytics) → 최고 정확도 & 속도
- DeepSORT (ReID 포함) ↔ SORT 자유 전환 가능
- RTMPose-m → TensorRT FP16 엔진으로 250+ FPS 키포인트 추정
- Redis 기반 추적 ID 영속성 (옵션)
- Docker + docker-compose로 1초 배포 가능

---

## 설치 방법 / 로컬 개발환경 구성 방법

### 로컬 실행 단계

1. **레포지토리 클론**
   ```bash
   git clone 

## 디렉토리와 파일의 구조 및 역할
```plaintext
hecto-ax-seed-portfolio/
├── code
│   ├── cfg/                      # config 파일
│   ├── utils/                    # 유틸리티 코드
│   │   ├── config_parser.py      # 설정 파일 파싱
│   │   └── logger.py             # 로깅 유틸리티
│   ├── model/                    # ai 모델 관련
│   │   ├── keypoint_estimator.py # keypoint estimator
│   │   └── trt_engine.py         # tensorRT
│   ├── pipeline/                 
│   │   └── stream_runner.py      # 
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
## 기능 추가, 수정, 유지보수 시 알아둘 것들
- trtexec --onnx=yolo11m.onnx --saveEngine=yolov11mfp32.engine



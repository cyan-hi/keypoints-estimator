import os
import subprocess

def build_yolo_engine(model="yolov11_m.pt", fp16=True, imgsz=640):
    model_path = f"weights/{model}"
    engine_name = model.replace(".pt", "_fp16.engine" if fp16 else ".engine")
    engine_path = f"weights/{engine_name}"

    if os.path.exists(engine_path):
        print(f"{engine_name} 이미 있음 → 스킵")
        return engine_path

    print(f"YOLO 엔진 빌드 중 → {engine_name} (FP16={'ON' if fp16 else 'OFF'})")
    
    cmd = [
        "yolo", "export",
        f"model={model_path}",
        "format=engine",
        f"half={'true' if fp16 else 'false'}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"성공! → {engine_name}")
        return engine_path
    else:
        print(f"실패! YOLO 엔진 생성 안 됨 → PyTorch로 자동 폴백")
        if result.stderr:
            print(result.stderr.splitlines()[-3:])  # 마지막 에러만 보여줌
        return None


def build_rtmpose_engine(onnx="rtmpose_m.onnx", fp16=True):
    onnx_path = f"weights/{onnx}"
    engine_name = onnx.replace(".onnx", "_fp16.engine" if fp16 else ".engine")
    engine_path = f"weights/{engine_name}"

    if os.path.exists(engine_path):
        print(f"{engine_name} 이미 있음 → 스킵")
        return engine_path

    print(f"RTMPose 엔진 빌드 중 → {engine_name}")
    
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}"
    ]
    if fp16:
        cmd.append("--fp16")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"성공! → {engine_name}")
        return engine_path
    else:
        print("실패")
        return None

if __name__ == "__main__":
    build_rtmpose_engine(onnx="rtmpose_m.onnx", fp16=True)
    # build_yolo_engine(model="yolov11_m.pt", fp16=True, imgsz=640) #pt로 inference시 실행 불필요
    print("\n모두 완료!")
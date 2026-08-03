# 03. 고급 운영·성능·보안

## 재현 가능한 실험

각 실행에서 다음 manifest를 보존합니다.

- OpenFabrik commit과 dirty 여부
- Python, CUDA, driver, PyTorch version
- model ID, revision·checkpoint hash와 license
- prompt·class file의 content hash
- CLI argument와 random seed
- session ID, 실패·retry 기록
- 생성·통과·제외 sample 수
- dataset tree hash와 split group 규칙

결과 file만 남기면 어떤 model·prompt·threshold로 만들었는지 재현하기 어렵습니다.

## 성능과 자원

- prompt, generation, annotation을 별도 queue로 분리하면 GPU 종류에 맞게 scheduling할 수 있습니다.
- model load를 image마다 반복하지 말고 worker lifetime 동안 재사용합니다.
- class 수·resolution·inference step별 VRAM peak와 처리량을 측정합니다.
- cache와 intermediate image에 quota·retention policy를 둡니다.
- OOM retry 전에 process가 GPU memory를 실제 반환했는지 확인합니다.

## Annotation calibration

작은 hand-labeled calibration set을 만들고 class별 precision·recall, mask IoU를 측정합니다. 하나의 global threshold가 모든 크기·class에 적합하다고 가정하지 않습니다. threshold를 dataset 자체에 맞추면 validation leakage가 생길 수 있으므로 calibration과 final test를 분리합니다.

## Domain gap 평가

다음 ablation을 비교합니다.

1. 실제 data만 학습
2. synthetic data만 학습
3. synthetic pretraining 후 실제 data fine-tuning
4. 실제 + synthetic 혼합 비율 변화
5. 생성 조건 또는 annotator별 제거 실험

Synthetic-only test 점수보다 실제 환경의 AP, class recall, calibration과 worst-group 성능을 우선합니다.

## 보안

- 외부 checkpoint는 code execution이 가능한 serialization 형식을 포함할 수 있으므로 출처·hash를 확인합니다.
- `trust_remote_code`가 필요한 model은 격리된 container와 최소 권한 계정에서 실행합니다.
- Ollama prompt에 비밀정보·고객 image metadata를 넣지 않습니다.
- 생성·reference image에 개인정보나 지적재산이 포함되는지 검토합니다.
- ROS2 node와 model service를 외부 network에 인증 없이 공개하지 않습니다.
- TensorRT/ONNX artifact도 공급망 provenance와 scan 대상에 포함합니다.

## Production 배포

YOLO → ONNX → TensorRT 변환 후 같은 validation set에서 output tolerance를 비교합니다. export 성공만으로 정확성 보존이 확인되는 것은 아닙니다.

운영에서 기록할 metric:

- end-to-end latency, preprocessing·inference·postprocessing 분해
- FPS보다 p50/p95/p99 latency
- GPU memory와 thermal throttling
- class별 confidence·detection count drift
- 입력 밝기·blur·resolution drift
- 사람 review와 false-negative escalation

## Architecture 확장

새 generator나 annotator를 추가할 때 pipeline 파일에 모든 logic을 넣지 말고 기존 module 경계를 따릅니다.

```text
pipeline orchestration
  → prompt/image generator interface
  → annotation interface
  → normalized intermediate record
  → YOLO writer
```

새 component는 작은 fixture에서 독립 실행, deterministic seed, failure cleanup, resume 동작과 output schema를 먼저 검증합니다.

## 기여 전 checklist

- public CLI argument와 documentation 동기화
- 함수·class docstring과 type hint
- CPU-only로 가능한 unit test와 GPU integration test 분리
- 작은 fixture로 label conversion 검증
- `pytest`, formatter/linter 결과 기록
- dependency·checkpoint·license 변경 명시
- 관련 없는 file reformat 제외

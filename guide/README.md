# OpenFabrik 한국어 학습 가이드

작성일: 2026-08-03
기준 commit: `3b58c971`

## 목표

이 가이드는 synthetic dataset 생성의 기본 개념부터 OpenFabrik 설치, 두 pipeline의 단계별 사용, YOLO dataset 검증, production 품질 관리와 확장까지 연결합니다. 전체 생성 pipeline은 대용량 GPU와 여러 외부 model이 필요하지만 `guide/examples/`의 검증 실습은 Python 표준 라이브러리만으로 실행할 수 있습니다.

## 학습 순서

1. [01. 설치와 첫 실행](01_getting_started.md)
2. [02. Pipeline workflow와 dataset 검수](02_pipeline_workflows.md)
3. [03. 고급 운영·성능·보안](03_advanced.md)
4. [GPU 없는 Python 실습](examples/README.md)

## 핵심 workflow

```text
문제 정의·class taxonomy
  → prompt/reference 설계
  → image·multi-view 생성
  → open-set/reference annotation
  → YOLO 구조 검증·사람 표본 검수
  → group-aware train/val split
  → YOLO 학습
  → 실제 촬영 test set 평가
  → 실패 분석 후 생성 분포·threshold 개선
```

Synthetic data의 목표는 많은 image를 만드는 것 자체가 아니라 target 환경에서 model error를 줄이는 것입니다. 수량, 시각적 다양성, annotation 정확도, 실제 환경과의 domain gap을 따로 측정해야 합니다.

## Pipeline 선택

| 질문 | Scene Generation | Reference Object |
| --- | --- | --- |
| 여러 일반 class를 text로 정의하는가? | 적합 | 제한적 |
| 특정 제품의 외형을 유지해야 하는가? | 제한적 | 적합 |
| 입력 image·mask가 있는가? | 불필요 | 필요 |
| 여러 시점이 중요한가? | prompt에 의존 | Qwen Multicamera / Zero123++ |
| 기본 annotator | SAM3 | PerSAM + SAM2 |

## 자원 계획

- CUDA GPU, 24GB 이상 VRAM
- model cache 최대 약 45GB
- prompt 생성용 Ollama service
- 생성 image와 intermediate session을 위한 별도 disk budget
- gated model 사용 전 license 동의와 Hugging Face 인증

처음에는 prompt 2~5개, image 1개씩의 smoke run으로 dependency와 output을 검증하세요. 전체 model을 받아 대량 생성하기 전에 `--list_sessions`, `--session_status`와 단계별 실행을 익히면 비용을 줄일 수 있습니다.

## 저장소 구조

| 경로 | 역할 |
| --- | --- |
| `pipelines/scene_generation_pipeline.py` | prompt→FLUX→SAM3/Grounded-SAM2 orchestration |
| `pipelines/reference_object_pipeline.py` | reference→multi-view→augmentation→annotation |
| `modules/diffusion_models/` | FLUX, Qwen image edit/multicamera, Zero123++ |
| `modules/open_set_models/` | SAM3, Grounding DINO + SAM2 |
| `modules/ref_segmentation/` | PerSAM + SAM2 reference segmentation |
| `modules/cv_processing/` | blur, compression, color·contrast augmentation |
| `utilities/yolo_scripts/` | split, statistics, curation, training, export |
| `utilities/ros2_scripts/` | sensor recording과 inference publish |

## 빠른 검증

OpenFabrik이 생성한 YOLO dataset에서 먼저 다음 명령을 실행합니다.

```bash
python guide/examples/01_validate_yolo_dataset.py ./dataset
python guide/examples/02_dataset_quality_report.py ./dataset --output quality-report.json
```

이 예제는 file pairing, label syntax, normalized coordinate, class range, split 중복과 class imbalance를 검사합니다. image 자체의 시각 품질과 mask 정확성은 별도 sample review가 필요합니다.

## 문서

- [한국어 README](../README_kor.md)
- [원본 README](../README.md)
- [공식 설치 문서](../docs/installation.md)
- [기여 안내](../CONTRIBUTING.md)
- [MIT License](../LICENSE)

다음 단계: [설치와 첫 실행](01_getting_started.md)

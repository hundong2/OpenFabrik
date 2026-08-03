<div align="center">

# 🏭 OpenFabrik

**데이터나 annotation 없이 computer vision model 시작하기**

*object detection과 segmentation을 위한 open-source synthetic data generation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[English](README.md) • [빠른 시작](#빠른-시작) • [Pipeline](#pipeline) • [예제](#예제) • [한국어 학습 가이드](guide/README.md)

</div>

---

## 문제

Computer vision model을 훈련하려면 수천 장의 label된 image가 필요합니다. 데이터 수집과 수동 annotation은 비용과 시간이 많이 들며 빠른 prototype과 실험의 병목이 됩니다.

## 해결 방법

OpenFabrik은 annotation이 포함된 synthetic training data를 자동 생성합니다. 탐지할 대상을 글이나 한 장의 reference image로 설명하면 YOLOv8, YOLOv10 및 현대적인 detection·segmentation model 학습에 사용할 수 있는 dataset을 만듭니다.

> “무제한”과 “완벽한 annotation”은 프로젝트가 지향하는 자동화 특성을 나타냅니다. 실제 synthetic image와 pseudo-label에는 생성 편향, 누락·오탐과 domain gap이 있을 수 있으므로 사람이 sample을 검수하고 실제 validation set으로 성능을 확인해야 합니다.

## 업데이트

2026-03-06 업데이트에서 Scene Generation Pipeline의 기본 annotator가 **SAM3**로 변경되었습니다. Promptable Concept Segmentation으로 여러 class를 처리하며, 24GB VRAM 환경을 위해 class별 순차 실행을 사용합니다. 기존 Grounding DINO + SAM2 조합은 `--annotator grounded_sam2`로 선택할 수 있습니다.

## 사용 사례

- **ML 연구**: 데이터 수집을 기다리지 않고 architecture를 빠르게 prototype
- **산업 vision**: 제조 품질 검사, defect detection, 재고 관리 dataset 준비
- **로보틱스**: manipulation·navigation을 위한 여러 시점의 object dataset 생성

OpenFabrik은 3D asset 없이 text 또는 사진 한 장으로 시작하고, open-set annotation과 YOLO bbox·segmentation output을 로컬에서 만드는 end-to-end pipeline을 목표로 합니다. 3D rendering framework보다 asset 준비가 적고 commercial synthetic-data service보다 제어권이 크지만, 정확한 기하·물리 ground truth가 중요한 경우 3D simulator가 더 적합할 수 있습니다.

## 주요 기능

- text description 또는 reference image에서 dataset 생성
- SAM3 또는 Grounding DINO + SAM2 기반 자동 annotation
- 배경·조명·시점 변형
- Qwen Multicamera 또는 Zero123++ 기반 multi-view 생성
- YOLO bbox·segmentation output
- YOLO 학습, ONNX·TensorRT export utility
- ROS2 실시간 inference 예제
- pipeline session 저장과 단계별 재시작

## 빠른 시작

### 요구 사항

- CUDA GPU와 **24GB 이상 VRAM**
- Python 3.8 이상(예시는 Python 3.10 환경 권장)
- 전체 pipeline model cache에 최대 약 **45GB** disk 공간
- prompt 생성을 위한 Ollama와 LLM
- SAM3 checkpoint는 Hugging Face license 동의와 인증이 필요할 수 있음

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull cogito:latest

conda create -n openfabrik python=3.10 -y
conda activate openfabrik

git clone https://github.com/cvar-vision-dl/OpenFabrik
cd OpenFabrik
pip install huggingface_hub
python utilities/download_models.py --cache_dir ./my_cache_dir --all
```

외부 install script는 조직 보안 정책에 따라 먼저 내용을 검토하세요. `--all`은 큰 model을 한꺼번에 받으므로 cache 여유 공간을 확인합니다.

### 설치

```bash
pip install -r requirements.txt
```

기본 SAM3 annotator를 사용하려면 repository 바깥에 SAM3를 clone하고 editable install합니다.

```bash
cd ..
git clone https://github.com/facebookresearch/sam3
cd sam3 && pip install -e . && cd ../OpenFabrik
```

Grounded-SAM2 fallback과 Reference Object Pipeline의 PerSAM은 README 원문의 external repository 설치 절차를 따릅니다. 외부 repository를 clone할 위치와 `PYTHONPATH`가 서로 다르므로 명령을 그대로 혼합하지 마세요.

### 첫 dataset 생성

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./my_dataset \
  --session new \
  --run_prompts --run_images --run_annotations \
  --project_info_file examples/prompts/kitchen_objects.txt \
  --predefined_classes "cup,bottle,glass,plate,spoon,knife,fork,bowl" \
  --num_prompts_per_execution 10 \
  --num_random_imgs 2 \
  --cache_dir ./my_cache_dir
```

결과는 session directory의 `outputs/<dataset_name>/` 아래 YOLO 구조로 생성됩니다. 학습 전에 [한국어 가이드](guide/README.md)의 dataset 검증 절차를 실행하세요.

```bash
yolo train data=./my_dataset/YYYYMMDD/outputs/dataset.yaml model=yolov8n.pt epochs=100
```

## Pipeline

### Scene Generation Pipeline

여러 object class를 가진 일반 detection dataset에 적합합니다.

1. Ollama LLM이 다양한 scene prompt 생성
2. FLUX diffusion model이 synthetic image 생성
3. SAM3가 text concept별 mask와 bbox 생성
4. class mapping과 YOLO dataset 저장

`--run_prompts`, `--run_images`, `--run_annotations`를 분리해 단계별로 실행할 수 있고 `--session last`로 이전 작업을 이어갈 수 있습니다. 기존 annotation stack이 필요하면 `--annotator grounded_sam2`를 사용합니다.

### Reference Object Pipeline

한 장의 reference image로 특정 제품·물체를 여러 시점에서 학습할 때 적합합니다.

1. reference image와 mask 준비
2. Qwen Multicamera(기본) 또는 Zero123++로 시점 생성
3. 배경·조명·가림 등 context augmentation
4. PerSAM + SAM2로 reference 기반 segmentation
5. generative·traditional CV augmentation

```bash
python pipelines/reference_object_pipeline.py \
  --input_image ./examples/pikachu_bag.jpg \
  --input_mask ./examples/pikachu_bag_mask.png \
  --project_info_file ./examples/prompts/project_pikachu.txt \
  --object_name "pikachu bag" \
  --num_prompts 10 \
  --num_iterations 1 \
  --working_dir ./datasets/my_product \
  --enable_annotation \
  --enable_qwen_augmentation \
  --qwen_augmentation_count 2 \
  --enable_cv_augmentation \
  --cv_augmentation_count 2 \
  --cache_dir ./my_cache_dir
```

## Pipeline 비교

| 항목 | Scene Generation | Reference Object |
| --- | --- | --- |
| 입력 | text description | reference image + mask |
| 주 용도 | multi-class 일반 dataset | 특정 object·제품 |
| 시점 | 기본적으로 single view scene | configurable multi-view |
| 생성 | FLUX | FLUX + Qwen Multicamera / Zero123++ |
| annotation | SAM3 기본, Grounded-SAM2 선택 | PerSAM + SAM2 |
| augmentation | 생성 prompt에 포함 | generative + CV |

## 예제

### 산업 부품 detection

```bash
python pipelines/scene_generation_pipeline.py \
  --predefined_classes "bolt,nut,washer,screw,gear" \
  --num_prompts_per_execution 100 \
  --num_random_imgs 5 \
  --working_dir ./datasets/industrial_parts \
  --cache_dir ./my_cache_dir \
  --session new --run_prompts --run_images --run_annotations
```

`--predefined_classes`는 현재 구현에서 comma-separated string으로 parsing되므로 공백으로 나눈 여러 argument가 아니라 위와 같이 전달합니다.

### 마지막 session의 annotation 재실행

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets/office_objects \
  --session last \
  --run_annotations \
  --cache_dir ./my_cache_dir
```

## Utility

- `utilities/download_models.py`: model cache 사전 다운로드
- `utilities/sam_mask_labeler.py`: reference mask 생성
- `utilities/yolo_scripts/yolo_training.py`: YOLO 학습
- `utilities/yolo_scripts/statistics_yolo_dataset.py`: dataset 통계
- `utilities/yolo_scripts/yolo_split_dataset.py`: train/validation 분할
- `utilities/yolo_scripts/yolo_export_onnx.py`: ONNX export
- `utilities/yolo_scripts/yolo_export_tensorrt.py`: TensorRT export
- `utilities/ros2_scripts/yolo_segmentation_publisher.py`: ROS2 inference publish

## Architecture

```text
Scene pipeline ─┬─ prompt_generator → FLUX → SAM3/Grounded-SAM2 ─┐
                │                                                ├→ YOLO dataset
Reference pipeline ─ image edit → multi-view → PerSAM/SAM2 ─────┘
                                      │
                               CV/generative augmentation
```

- `pipelines/`: end-to-end orchestration과 session 관리
- `modules/llm_models/`: prompt 생성
- `modules/diffusion_models/`: FLUX, Qwen, Zero123++ image 생성
- `modules/open_set_models/`: SAM3와 Grounded-SAM2 annotation
- `modules/ref_segmentation/`: PerSAM reference segmentation
- `modules/cv_processing/`: 전통적 image augmentation
- `utilities/`: model download, dataset, export, ROS2 도구

## 품질과 안전

- synthetic-only 성능이 아니라 실제 촬영한 고정 validation/test set에서 평가합니다.
- class prompt와 annotator threshold별 누락·오탐을 sample audit합니다.
- 같은 base image에서 파생된 augmentation이 train과 validation에 나뉘지 않게 group split합니다.
- 사람·상표·민감 시설 등 생성 content의 법적·윤리적 제한을 확인합니다.
- 외부 model license, gated checkpoint 조건과 상업적 사용 가능 여부를 각각 확인합니다.
- `torch.load`, pickle, 외부 checkpoint와 editable dependency는 신뢰할 수 있는 출처만 사용합니다.

## 문서와 학습 자료

- [한국어 학습 가이드](guide/README.md)
- [설치 문서](docs/installation.md)
- [Scene Generation Pipeline](docs/scene_generation_pipeline.md)
- [Reference Object Pipeline](docs/reference_object_pipeline.md)
- [Configuration](docs/configuration.md)
- [기여 안내](CONTRIBUTING.md)

## 인용과 License

연구에 사용할 때는 원본 README의 BibTeX를 참고하세요. OpenFabrik은 [MIT License](LICENSE)로 배포되며, 결합하는 model·dataset·외부 repository에는 각각 별도 license가 적용됩니다.

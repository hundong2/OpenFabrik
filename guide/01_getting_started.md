# 01. 설치와 첫 실행

## 목표

실행에 필요한 GPU·disk·외부 repository를 준비하고, 작은 Scene Generation smoke run을 단계별로 수행합니다.

## 1. 환경 점검

```bash
nvidia-smi
python --version
df -h
```

권장 기준은 Python 3.10, CUDA GPU 24GB 이상 VRAM과 model cache 45GB 이상의 여유 공간입니다. CUDA·PyTorch 조합은 `requirements.txt`의 `cu126` index를 그대로 사용하기 전에 host driver와 맞는지 확인하세요.

## 2. 격리 환경

```bash
conda create -n openfabrik python=3.10 -y
conda activate openfabrik
pip install -r requirements.txt
```

운영이나 반복 실험에서는 설치 후 `pip freeze` 또는 lock 도구로 실제 dependency를 기록합니다. `requirements.txt`는 lower bound 중심이므로 미래 package 조합이 동일하게 동작한다는 보장은 없습니다.

## 3. 외부 component

Scene pipeline 기본값은 SAM3입니다. SAM3는 repository 밖에 설치합니다.

```bash
cd ..
git clone https://github.com/facebookresearch/sam3
cd sam3
pip install -e .
cd ../OpenFabrik
```

Grounded-SAM2 fallback은 OpenFabrik 안의 `Grounded-SAM-2/` 경로를 기대하는 기본 argument가 있고, PerSAM은 별도 repository·`PYTHONPATH` 설정이 필요합니다. 한꺼번에 설치하지 말고 사용할 pipeline에 필요한 component만 준비하세요.

## 4. Ollama와 model cache

```bash
ollama pull cogito:latest
python utilities/download_models.py --cache_dir ./my_cache_dir --all
```

`--all` 대신 필요한 model flag만 선택하면 download와 disk 사용량을 줄일 수 있습니다. checkpoint hash, source URL, license 승인 날짜를 실험 manifest에 기록하세요.

## 5. 작은 smoke run

먼저 prompt만 생성합니다.

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./runs/smoke \
  --session new \
  --run_prompts \
  --project_info_file examples/prompts/kitchen_objects.txt \
  --predefined_classes "cup,bottle" \
  --num_prompts_per_execution 2 \
  --cache_dir ./my_cache_dir
```

생성된 prompt를 사람이 검토한 후 같은 session에서 image와 annotation을 이어갑니다.

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./runs/smoke \
  --session last \
  --run_images --run_annotations \
  --num_random_imgs 1 \
  --cache_dir ./my_cache_dir
```

단계 분리는 prompt 오류를 대량 image 생성 전에 발견하고, OOM이나 annotation 실패가 나도 이전 산출물을 재사용하게 해 줍니다.

## 6. 결과 확인

```text
<working_dir>/<session>/
  prompts 또는 중간 metadata
  generated images
  outputs/<dataset_name>/
    images/train/
    labels/train/
    dataset.yaml
    classes.txt
```

실제 session 이름과 세부 경로는 실행 log와 session metadata를 기준으로 확인하세요.

```bash
python guide/examples/01_validate_yolo_dataset.py \
  ./runs/smoke/<session>/outputs/generated_dataset
```

## 흔한 오류

### CUDA out of memory

다른 GPU process를 종료하고 image resolution·batch성 작업량을 낮춥니다. SAM3는 class를 순차 처리하지만 class 수와 image 수에 따라 총 시간은 늘어납니다.

### SAM3 checkpoint 접근 실패

Hugging Face model page에서 license에 동의하고 `hf auth login`을 완료했는지 확인합니다. cache directory가 실행 계정에 읽기 가능한지도 검사합니다.

### Ollama 연결 실패

Ollama service 상태, model 이름과 port를 확인합니다. 기존 prompt file을 재사용할 수 있다면 image·annotation 단계만 실행해 LLM dependency를 분리합니다.

### annotation이 비어 있음

class 표현이 image prompt와 annotator concept에 일치하는지 확인하고 threshold를 무작정 낮추기 전에 image에 대상이 실제로 있는지 봅니다. sample별 false positive/negative audit를 남깁니다.

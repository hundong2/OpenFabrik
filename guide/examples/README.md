# GPU 없는 dataset 검증 실습

OpenFabrik 전체 pipeline을 실행하지 않고 생성된 YOLO dataset의 구조와 통계를 검사하는 Python 표준 라이브러리 예제입니다.

## 1. 구조와 label 검증

```bash
python guide/examples/01_validate_yolo_dataset.py ./path/to/dataset
```

검사 항목:

- `images/`와 `labels/`의 stem pairing
- detection·segmentation label token 수
- class ID와 normalized coordinate 범위
- 빈 label과 누락된 image·label
- train/val의 동일 byte image leakage

오류가 있으면 exit code `1`, 정상이면 `0`을 반환하므로 CI gate로 사용할 수 있습니다.

## 2. 품질 통계 report

```bash
python guide/examples/02_dataset_quality_report.py \
  ./path/to/dataset \
  --output quality-report.json
```

class별 instance 수, 빈 image, image당 annotation 분포와 imbalance ratio를 JSON으로 저장합니다. class 이름은 `classes.txt`가 있으면 읽고, 없으면 숫자 ID를 사용합니다.

## 지원 구조

```text
dataset/
  images/train/*
  images/val/*
  labels/train/*.txt
  labels/val/*.txt
```

flat `images/*`, `labels/*`도 지원합니다. Image decode, perceptual duplicate, mask visual quality는 검사하지 않으므로 OpenCV/Pillow 기반 검수와 사람 sample review를 추가하세요.

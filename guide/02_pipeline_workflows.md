# 02. Pipeline workflow와 dataset 검수

## 목표

두 pipeline의 단계와 실패 지점을 이해하고 synthetic dataset을 학습에 넣기 전 품질 gate를 설계합니다.

## Scene Generation 흐름

### Prompt

`project_info_file`은 작업 목적, 대상, 환경과 원하는 다양성을 설명합니다. `predefined_classes`를 사용하면 dataset 전체 class ID를 안정적으로 고정할 수 있습니다.

좋은 prompt taxonomy는 다음을 분리합니다.

- class identity: cup, bottle
- context: kitchen, warehouse
- nuisance factor: 조명, 가림, motion blur
- viewpoint·scale: top-down, close-up, distant
- negative scene: 대상 class가 없는 배경

### Image generation

FLUX output의 다양성을 늘리되 class 모양이 무너지거나 text artifact가 label shortcut이 되지 않는지 확인합니다. 단순 prompt 개수보다 viewpoint·background·scale bin별 coverage를 측정합니다.

### Annotation

SAM3는 predefined concept를 class별로 처리합니다. Grounded-SAM2 fallback에서는 text detection threshold와 segmentation이 별도 실패할 수 있습니다. bbox가 있어도 mask edge가 부정확할 수 있으므로 두 품질을 나눠 봅니다.

## Reference Object 흐름

Reference image와 mask의 품질이 모든 파생 view에 영향을 줍니다. 배경이 깨끗하고 object 전체가 보이는 image를 사용하고 mask의 hole·잘린 경계를 먼저 고칩니다.

Qwen Multicamera는 configurable view를 생성하고 Zero123++는 대안적인 6-view 경로입니다. 생성 view에서 logo, handle, connector처럼 identity를 결정하는 부분이 변형되지 않았는지 확인합니다.

Generative augmentation과 CV augmentation을 동시에 많이 적용하면 실제 camera 분포보다 과도한 degradation이 생길 수 있습니다. 각 augmentation의 확률·범위를 production sensor 측정치에 맞춥니다.

## YOLO label 검수

Detection label 한 줄은 일반적으로 다음 형식입니다.

```text
class_id x_center y_center width height
```

Segmentation은 `class_id` 뒤에 normalized polygon point가 이어집니다. 모든 좌표는 보통 `[0,1]` 범위여야 하고 polygon에는 최소 3개 point가 필요합니다.

필수 자동 gate:

- image와 label stem pairing
- 읽을 수 없는 빈/손상 file
- class ID 범위와 class mapping 안정성
- bbox·polygon coordinate 범위와 면적
- image당 object 수 분포
- class별 instance·image 수
- 동일·유사 image의 train/val leakage

필수 사람 gate:

- class 정확성, 누락·오탐
- mask boundary와 작은 object
- 비현실적 geometry·shadow·reflection
- 생성 model watermark/text shortcut
- 실제 배치 조건과 viewpoint·scale 차이

## Split 전략

한 base image에서 파생한 multi-view·augmentation은 같은 group으로 묶어 한 split에만 넣습니다. 무작위 file split은 거의 같은 image가 train과 validation에 들어가 점수를 부풀릴 수 있습니다.

권장 계층:

```text
object identity / source prompt / base image
  └─ views
      └─ generative variants
          └─ CV augmentations
```

가장 상위 group ID를 기준으로 split합니다. 실제 촬영 test set은 synthetic generation 과정과 완전히 분리합니다.

## 반복 개선

1. 작은 synthetic dataset을 생성합니다.
2. 자동·사람 품질 gate를 통과시킵니다.
3. 작은 YOLO model로 빠르게 학습합니다.
4. 실제 validation에서 class·조건별 실패를 수집합니다.
5. 부족한 condition만 prompt와 augmentation에 추가합니다.
6. 고정 test set은 반복 중 건드리지 않습니다.

더 많은 synthetic sample이 항상 개선을 의미하지 않습니다. 실제 error taxonomy를 생성 policy의 입력으로 되돌리는 closed loop가 중요합니다.

# Vast.ai Jupyter 시작 가이드

## NATTEN 설치 중 compute_1200 오류가 난 경우

이전 스크립트에서 compute capability `(12, 0)`을 `120`으로 전달하는 오류가 있었다.
NATTEN은 점이 있는 `12.0`을 요구하며, `120`을 전달하면 `compute_1200`으로 변환한다.
이 오류는 스크립트에서 수정했고, 잘못된 환경변수도 빌드 전에 검증한다.

이미 NATTEN 이전 의존성 설치가 끝난 서버에서는 저장소 루트에서 다음 명령으로 재개한다.

```bash
git pull --ff-only
NATTEN_CUDA_ARCH=12.0 bash scripts/setup_vast.sh --resume-natten
```

새 로그는 `outputs/setup/natten-build.log`에 저장된다. 이 수정은 아키텍처 표기 오류를 해결하며,
이후 실제 컴파일·CUDA 연산 성공 여부는 해당 서버에서 확인한다.

## 5090 고품질 첫 실행

노트북의 손상 렌더 단계는 `configs/vast/render_high_quality.json`을 사용한다.
CUDA / Cycles / 1024×1024 / 128 samples로 정상·찌그러짐·찍힘·긁힘 각 1장, 총 4장을 먼저 생성한다.
이 설정은 실행 준비용이며 실제 5090에서의 시간·VRAM·품질 검증은 아직 필요하다.
4장과 라벨을 확인한 후 설정 파일의 `count`를 늘리고 새 출력 폴더를 지정한다.
기존 `render.json`의 CPU·768px·32 samples 설정은 로컬 시험용으로 유지한다.

## 실행 흐름

공개 GitHub 저장소 → Vast.ai Jupyter 터미널에서 clone → 의존성 설치 → 모델 준비 → 범퍼 생성·검수 → 손상·RGB·마스크·YOLO 데이터.

아래 공개 저장소 주소로 clone한다. 개인 서버 접속 주소나 인증값은 저장소에 포함하지 않는다. 서버 대여·과금·종료는 이 코드가 수행하지 않는다.

```bash
git clone https://github.com/biop99999111/Pixal3D-vast-bumper.git
cd Pixal3D-vast-bumper
bash scripts/setup_vast.sh
```

설치 후 `notebooks/workspace.ipynb`를 열고 **Pixal3D Vast** 커널을 선택한다. 기본 코드와 이미지·모델 캐시는 같은 서버에 두고 작업한다.

## 1. 서버 준비

- Linux, Python 3.10 이상, 호환되는 CUDA PyTorch/torchvision, CUDA 개발 도구 `nvcc`, C++ 컴파일러, git이 필요하다.
- RTX 5090에서는 CUDA 12.8 이상에 대응하는 PyTorch 및 확장 패키지 조합을 선택한다. CUDA runtime-only 이미지에는 컴파일 도구가 없을 수 있다.
- 설치 스크립트는 기존 Torch를 유지하고 의존성 제약으로 기록한다. 임의로 구형 Torch로 낮추지 않는다.
- `configs/vast/source-refs.json`에 고정된 TRELLIS.2, CuMesh, FlexGEMM, MoGe 및 nvdiffrast를 설치한다. NATTEN은 원본 기준 0.21.0이다.
- 이 설치 조합의 실제 5090 성공 여부는 아직 확인되지 않았다. `check_environment.py --gpu-smoke`의 실제 CUDA 연산 검사와 단일 이미지 추론이 서버에서 통과해야 한다.
- 디스크에는 모델, 빌드 캐시, `mesh_state.npz`, 출력 텍스처와 렌더링 데이터가 저장된다. 진단의 여유 공간을 확인한다.

```bash
python scripts/check_environment.py --gpu-smoke
```

`outputs/environment.json`에 패키지 import, CUDA·GPU 정보, SDPA·NATTEN·nvdiffrast 최소 연산 결과를 기록한다. CuMesh·FlexGEMM·o_voxel의 전체 경로는 실제 생성·export에서 추가 검증된다.

Blender는 별도 설치한 실행 파일이 필요하다. 로컬 검증은 Blender 5.2.1 LTS를 사용했다. 서버에서도 같은 버전 사용을 우선 권장하며, `blender --version`과 작은 fixture 렌더로 확인한다. 원하는 경로의 Blender를 `--blender /path/to/blender`로 지정할 수 있다. RGB 렌더의 `device`는 CPU, CUDA, OPTIX 중 사용 환경에 맞게 설정한다. GPU가 없으면 조용히 CPU로 전환하지 않는다.

## 2. 모델과 인증

기본은 `provided_mask`: 범퍼만 남은 투명 PNG 또는 외부 그레이스케일 마스크를 사용한다. 배경 제거 모델 생성·다운로드·추론을 생략한다. RGB 자동 제거는 `auto`로 선택한다.

```bash
python scripts/prepare_models.py --background-mode provided_mask
# RGB 이미지의 자동 배경 제거가 필요하면:
python scripts/prepare_models.py --background-mode auto
```

필요한 인증은 notebook의 숨김 입력으로 `HF_TOKEN`을 전달한다. 토큰을 코드·명령문·설정 파일에 직접 적지 않는다. 공개 모델도 실제 접근 조건은 다운로드 시 확인한다. 모델 페이지의 접근 승인이 필요한 경우 인증만으로 승인 절차가 대체되지는 않는다.

모델 준비 결과:

- `cache/models/models.lock.json`: HF 저장소 revision, 로컬 모델 경로, NAF 커밋.
- `cache/models/pipeline/pipeline.json`: 로컬 체크포인트에 연결된 실행 설정.
- NAF는 Torch Hub 경로로 CPU에서 준비하며 별도 캐시를 사용한다.
- lock의 절대 경로는 서버별로 다르므로 로컬 PC의 lock을 서버에 그대로 복사하지 않는다.
- 같은 출력 폴더에서는 lock의 revision을 재사용한다. 의도적으로 갱신할 때만 `--refresh`를 사용한다.

## 3. 범퍼 생성

예제 사진은 `assets/images/bumper.jpg`에 포함되어 있다. 노트북에서 업로드 없이 실행하면 이 사진을 사용한다. 배경이 있는 JPEG이므로 `auto` 모드를 선택하거나 별도 마스크를 준비한다.

`configs/vast/baseline.json`을 복사하여 실험 설정을 만든다. 초기 기본값은 표준 모드, 생성 해상도 1536, 단계별 12 steps, PNG 계열 GLB 출력이다.

```bash
python -m bumper_synth infer \
  --image inputs/bumper.png \
  --config configs/vast/baseline.json \
  --models-lock cache/models/models.lock.json \
  --output outputs/bumper_001
```

외부 마스크 사용 시 `--mask inputs/bumper_mask.png`를 추가한다. 마스크는 원본 이미지와 크기가 같아야 하며 흰색이 범퍼, 검정이 배경이다. 작은 구멍도 검정으로 보존한다. RGB 자동 제거 시에는 config의 `background_mode`와 준비한 lock을 모두 `auto`로 맞춘다.

주요 산출물:

| 파일 | 내용 |
|---|---|
| `model_input.png`, `input_mask.png` | 실제 모델 입력과 전경 마스크 |
| `foreground_rgba.png`, `preprocess.json` | 배경 분리 결과와 크롭/리사이즈 변환 |
| `camera.json` | 추정/수동 카메라와 GLB export 변환 |
| `stages.json` | 모델 로딩·전처리·형상·텍스처·export 시간 및 Torch peak VRAM |
| `mesh_state.npz`, `.layout.json` | 생성 재실행 없이 출력 옵션을 바꾸기 위한 중간 표현 |
| `bumper.glb`, `quality.json`, `textures/` | 3D 자산, 구조 점검, 텍스처 추출 |
| `run.json` | 입력·설정·소스·모델 식별자와 성공/실패 상태 |

실패 시 low VRAM이나 낮은 해상도로 자동 전환하지 않는다. 결과 폴더는 입력/설정/소스가 같을 때만 재사용한다. 설정을 바꾸면 새 출력 폴더를 사용한다. 중단된 추론의 중간 확산 단계에서 이어가기 대신 해당 실행을 처음부터 재실행한다.

## 4. 범퍼·GLB 검수

```bash
python -m bumper_synth preview --asset outputs/bumper_001/bumper.glb --output outputs/bumper_001/preview
python -m bumper_synth align --run outputs/bumper_001
```

- preview: 4방향 회전 이미지. 입력 밖의 차체·바퀴 생성 여부, 뒤집힘, 가짜 손상을 확인한다.
- align: 입력 카메라 및 좌표 변환으로 Base Color, 실루엣, 노멀을 렌더링한다. `overlay.png`, `difference.png`, 실루엣 IoU·전경 RGB MAE를 기록한다.
- Base Color와 사진의 조명 차이가 있으므로 RGB MAE 하나로 품질 합격을 결정하지 않는다.
- 범퍼 의미 판정이나 뒷면 복원 정확성을 자동으로 보장하는 기능은 없다. 실패 자산은 검수에서 제외하고 추가 사진 또는 검수된 다른 메시를 사용한다.
- 수동 GLB는 `render.json`의 `rotation_degrees`, `front_axis`를 맞춘다. 생성 폴더의 camera.json이 있으면 입력 카메라 기준으로 자동 정렬한다.

Khronos 공식 Validator도 선택적으로 실행할 수 있다(Node.js 필요).

```bash
npm ci --ignore-scripts
node scripts/validate_glb.mjs outputs/bumper_001/bumper.glb outputs/bumper_001/gltf-validation.json
```

실제로 검수한 후 기록한다. 이것은 사용자가 검수 사실을 기록하는 명령이다.

```bash
python -m bumper_synth review --asset outputs/bumper_001/bumper.glb \
  --note '범퍼 단독 형상, 깨끗한 표면, 정면과 제한적 사선 시점 검수 완료'
```

기록은 GLB 해시에 묶인다. 파일이 변경되면 재검수가 필요하다.

## 5. 손상 데이터 생성

`configs/vast/render.json`을 복사한다. `count`, `resolution`, `samples`, `device`, 손상 크기·깊이, 카메라 각도를 조절한다.

```bash
python -m bumper_synth render \
  --asset outputs/bumper_001/bumper.glb \
  --config configs/vast/render.json \
  --output outputs/dataset_part001 \
  --group part001 --split train
```

- 0 `dent`: 넓은 국소 메시 함몰(찌그러짐).
- 1 `ding`: 작은 국소 함몰(찍힘). 도장 벗겨짐의 별도 모델링은 현재 포함하지 않는다.
- 2 `scratch`: 유한 길이의 표면 경로에 Base Color·거칠기·bump 변경(긁힘).
- 초기 구현은 이미지당 손상 1개이며 정상 이미지도 포함한다. 복합 손상·충돌 물리 시뮬레이션은 지원하지 않는다.
- 손상 단위는 정규화 자산 최대 길이 2에 대한 상대값이다. 실제 mm를 의미하지 않는다.
- 전면 ray cast로 위치를 선택한다. 메시가 너무 성기면 오류를 내므로 밀도 높은 기본 자산을 사용한다.
- RGB와 마스크는 같은 카메라·변형 메시를 사용한다. 마스크는 손상 영향 표면의 가시 영역이며 그림자 자체를 포함하지 않는다.
- 최소 마스크 픽셀·박스 크기와, 같은 장면의 무손상 렌더 대비 손상 내부 평균 색상 변화가 기준 미만이면 다른 위치로 재시도한다. 이 수치 검사는 실제 식별 가능성의 완전한 대체가 아니다.
- 단일 폴더 재실행 시 완료 샘플의 파일 해시를 검사하여 재사용한다. 실패 로그는 `work/`에 남긴다.

출력은 `images/<split>/`, `labels/<split>/`, `masks/`, `previews/`, `metadata/`, `dataset.json`이다. 마스크는 손상 인스턴스 PNG, 라벨은 YOLO 검출 박스 형식이다. 세그멘테이션 다각형 변환은 아직 제공하지 않는다.

## 6. 데이터 결합과 학습

`--group`은 원본 부품/차량/촬영 그룹이다. 같은 부품의 다른 시드·시점·재텍스처 결과에도 같은 그룹을 사용한다. 하나의 범퍼만 있다면 train/val 평가 분할을 만들지 않고 우선 생성 기능만 검증한다.

서로 다른 원본 범퍼로 train과 val 데이터를 만든 후 결합한다.

```bash
python -m bumper_synth assemble \
  --datasets outputs/dataset_part001 outputs/dataset_part002_val \
  --output outputs/yolo_dataset
```

동일 그룹 또는 동일 GLB가 다른 split에 나타나거나, 파일 해시가 바뀌었거나, 시험용 fixture 데이터면 결합을 거부한다. 결합 결과에 `data.yaml`을 만든다.

학습은 Pixal3D 의존성과 충돌하지 않도록 별도 Python 환경에 Ultralytics를 설치하고 사용한다. 사용할 검출 모델 체크포인트는 `--model`로 지정한다.

```bash
python scripts/train_yolo.py --data outputs/yolo_dataset/data.yaml \
  --model YOUR_DETECTION_CHECKPOINT.pt --epochs 50 --imgsz 768
```

실제 라벨 데이터가 있으면 `--real-eval-data /path/to/real_data.yaml`을 추가한다(test split 필요). 실제-only와 실제+합성 실험을 같은 모델·seed·학습 조건으로 비교한다. 실제 사진 평가가 없으면 합성데이터의 현장 효과를 판단할 수 없다.

## 7. 출력 옵션과 파라미터 비교

생성을 다시 하지 않고 GLB 텍스처 크기·면 수·remesh·WebP를 변경한다.

```bash
python -m bumper_synth export --state outputs/bumper_001/mesh_state.npz \
  --config configs/vast/baseline.json --output outputs/export_2k.glb
```

해상도와 메모리 운용 방식 비교:

```bash
python scripts/run_experiments.py --image inputs/bumper.png \
  --models-lock cache/models/models.lock.json --output outputs/comparison --steps-sweep
```

각 실행은 별도 프로세스다. 표준 1024와 low VRAM 1024를 먼저 비교하고, 표준 1536의 효과를 별도로 본다. summary.csv는 시간·VRAM 요약이며 품질 순위를 자동 결정하지 않는다.

## 8. 실험적 Stable Diffusion 텍스처 보정

`align` 결과를 바탕으로 깨끗한 기본 부품의 정면 Base Color를 inpaint하고, 원래 UV에 투영·베이킹하여 새 GLB를 만든다. Diffusers 호환 inpainting 모델 ID와 프롬프트가 필요하다.

```bash
python -m bumper_synth retexture --run outputs/bumper_001 \
  --output outputs/bumper_refined --model YOUR_INPAINTING_MODEL \
  --prompt 'clean undamaged painted bumper surface, preserve original color and details' \
  --strength 0.25
```

메시 형상과 기존 다른 재질 채널을 유지하며 가려진 표면은 기존 색상을 사용한다. 정면 단일 시점 방식으로, 다중 시점 일관성 최적화는 없다. 경계·가짜 손상·조명이 텍스처에 남는지 검수한다. 결과는 `bumper_refined.glb`이며 검수 기록을 자동 승계하지 않는다. 라벨이 완성된 손상 RGB에 이 기능을 적용하지 않는다.

Blender 베이킹 경로는 로컬에서 시험했지만 실제 SD 모델 다운로드·추론과 실제 범퍼 외관 개선은 미검증이다.

## 9. 모델 다운로드 없이 설치 확인

```bash
python -m bumper_synth fixture --output outputs/fixture/panel.glb
python -m bumper_synth render --asset outputs/fixture/panel.glb \
  --config configs/vast/render.json --output outputs/fixture/dataset --group fixture --demo
```

fixture는 기능 검증용 곡면 패널이며 범퍼 복원 결과나 학습데이터가 아니다.

## 10. 공개 저장소 준비

```bash
python -m pytest tests/test_bumper_synth.py -q
python scripts/build_notebook.py
python scripts/package_source.py
```

`dist/Pixal3D-vast-bumper-source.zip`에는 회사 입력, 결과, 가중치, 캐시, 토큰, notebook 출력·위젯 상태를 포함하지 않는다. 원본 LICENSE/NOTICE는 유지한다. GitHub 게시 전 포함 파일을 확인한다. 업로드된 저장소에서 위 clone 흐름을 사용하면 된다.

## 참고

- Vast.ai Jupyter: https://docs.vast.ai/guides/instances/connect/jupyter
- NATTEN 설치/Blackwell: https://natten.org/install/
- TRELLIS.2 설치 기준: https://github.com/microsoft/TRELLIS.2
- YOLO 검출 형식: https://docs.ultralytics.com/datasets/detect
- glTF Validator: https://github.com/KhronosGroup/glTF-Validator

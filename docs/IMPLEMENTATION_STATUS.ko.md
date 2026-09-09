# 구현 및 검증 현황

2026-09-09, 제공된 Pixal3D-source.zip 기준.

## 구현한 기능

| 기능 | 코드 |
|---|---|
| Vast.ai 설치, 의존성 커밋 고정, 기존 Torch 유지 | `scripts/setup_vast.sh`, `configs/vast/source-refs.json` |
| GPU/패키지 진단 및 실제 CUDA 최소 연산 검사 | `scripts/check_environment.py` |
| HF 모델 다운로드·revision 기록, NAF 준비 | `scripts/prepare_models.py` |
| 제공 마스크 모드에서 배경 제거 모델 로딩까지 생략 | 원본 pipeline의 `load_rembg` 옵션 |
| 입력·마스크·크롭 변환 보존, 빈 마스크 검증 | `bumper_synth/preprocess.py` |
| 설정 기반 추론, OOM 실패 기록, 단계별 Torch VRAM·시간 | `inference.py`, `telemetry.py` |
| 중간 메시 저장 및 출력 옵션별 GLB 재출력 | `export.py` |
| GLB 구조/텍스처 점검, 4방향 미리보기 | `quality.py`, `preview.py` |
| 입력 카메라 렌더, Base Color·노멀·실루엣 비교 | `alignment.py`, `blender_worker.py` |
| GLB 해시에 묶인 사용자 검수 기록 | `quality.py` |
| 찌그러짐·찍힘 메시 변형, 긁힘 재질/bump 생성 | `blender_worker.py` |
| RGB·가시 손상 마스크·YOLO 박스·미리보기·메타데이터 | `dataset.py`, `labels.py` |
| 정상 샘플, 무손상 대비 변화량 검사, 완료 샘플 재사용 | `dataset.py` |
| 원본 그룹/동일 자산의 split 누출 검사 | `dataset.py assemble` |
| 해상도/모드/steps 실험과 CSV 요약 | `scripts/run_experiments.py` |
| YOLO 학습·선택적 실제 test 평가 인터페이스 | `scripts/train_yolo.py` |
| 정면 SD inpainting + UV 투영·베이킹(실험 기능) | `retexture.py`, `blender_worker.py` |
| 업로드·숨김 인증·설정·검수·다운로드 notebook | `notebooks/workspace.ipynb` |
| 공개 소스 ZIP 생성 | `scripts/package_source.py` |
| Khronos 공식 glTF 검증 | `scripts/validate_glb.mjs` |

위 모듈명은 별도 표기가 없으면 `bumper_synth/` 아래에 있다. 기존 `app.py`, `inference_mv.py`는 원본 동작을 유지하며, 공통 설정은 새 `python -m bumper_synth` 및 notebook 경로에서 제공한다.

## 실제 로컬 검증

- CPU 단위 테스트 7개 통과: 알파/구멍 보존, 빈 마스크·불투명 입력 거부, 화면 경계 라벨, 자산 변경 시 검수 무효화, 그룹 누출·fixture 거부, 카메라 투영 변환, 잘못된 YOLO 좌표 거부.
- Python 모듈 및 notebook 셀 구문 검사, 설치 Bash 구문 검사.
- Blender 5.2.1 LTS에서 시험용 곡면 패널 GLB 생성.
- 정상, 찌그러짐, 찍힘, 긁힘의 실제 RGB·마스크·YOLO 박스 생성 및 이미지 확인.
- 동일 카메라 Base Color·실루엣·노멀 렌더 성공.
- 이미지 투영→UV 베이킹→GLB 재출력 성공.
- 베이킹 전후 정점 4,141개, 삼각형 8,000개 유지. 정점 좌표와 면 배열 동일 확인.
- 시험 패널 및 베이킹 GLB는 Khronos Validator에서 각각 오류 0개, 경고 0개.

시험 패널은 범퍼 복원 결과가 아니다. 위 검증은 소프트웨어 연결과 라벨·베이킹 경로를 확인한 것이다.

## 실제 서버/모델 검증이 필요한 항목

- Vast.ai 5090 전체 설치 및 Pixal3D 모델 다운로드·추론·export.
- 실제 범퍼에서 자동차 전체가 생성되는 문제의 개선 여부.
- 실제 재질의 손상 사실감, 촬영 범위, 찍힘 클래스 정의.
- SD inpainting 모델 다운로드·추론·품질 개선. Blender 베이킹 경로만 별도 실행 검증했다.
- YOLO 학습 및 독립 실제 사진 평가.

## 초기 버전 범위

- 이미지당 손상 1개. 복합 손상·충돌 물리 해석·실측 치수 복원은 지원하지 않는다.
- YOLO 검출 박스와 PNG 마스크 제공. 세그멘테이션 다각형 변환은 미포함이다.
- 표면·조명·단색 배경 변화를 제공한다. 차량 장착 장면 합성에는 별도 자산이 필요하다.
- 텍스처 보정은 정면 한 시점 기준이며 다중 시점 최적화는 아니다.
- 부품 의미 검수는 사용자가 수행한다. 모델을 범퍼 전용으로 재학습하거나 의미를 강제하는 기능은 없다.
- 샘플 단위 재시작을 지원하지만 Pixal3D 확산 중간 단계에서 재개하지는 않는다.
- 자동 마스크는 그림자가 아니라 수정된 표면 영역이다. 실제 어노테이션 정책에 맞는지 검수가 필요하다.

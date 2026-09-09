# 범퍼 공간 제약 실험

예제 `assets/images/bumper.jpg`와 이전 `model_input.png`는 이미 범퍼만 포함한다.
자동 배경 제거가 자동차를 남긴 문제가 아니라 단일 이미지 생성 모델이 보이지 않는
차체까지 생성한 사례다. 마스크만 바꾸면 해결된다고 보장할 수 없다.

`--bumper-only`는 기존 생성 모델에 실루엣과 깊이 범위를 이용한 좌표 제약을 추가한다.
1536, low_vram=false, 기존 시드와 샘플러를 유지한다. 희소 구조 생성 직후와
고해상도 좌표 생성 후, 이미지의 외곽 실루엣 밖 및 지정 깊이 밖의 좌표를 제거한다.
그릴 내부의 구멍은 좌표 선택용 실루엣에서만 채운다. 실제 입력 알파와 색상은 유지한다.

예제 사진은 확인된 이전 실행의 foreground_rgba.png에서 알파를 재사용한다.
다른 사진에는 원본 크기의 범퍼 전용 마스크를 --mask로 지정해야 한다.
마스크를 자동으로 범퍼라고 판별하는 기능은 아니다.

```bash
cd /workspace/Pixal3D-vast-bumper
git pull --ff-only origin fix/sparse-projection-1536
python scripts/run_tuned_1536.py \
  --previous outputs/bumper_1536_sparse_001/run.json \
  --bumper-only \
  --output outputs/bumper_only_1536_001
```

기본 깊이 범위는 생성 좌표계 z=0.15~0.5이다. 카메라 쪽 전면에 해당하며
물리적인 미터 단위나 측정한 범퍼 두께가 아니다. --depth-min과 --depth-max로
조절할 수 있다. 범위를 좁힐수록 옆면/장착부도 잘릴 수 있다.
전면 표면을 추정하도록 유도하는 실험이며 정확한 부품 복원이나 텍스처 개선을 보장하지 않는다.
디코더 및 리메시 단계는 좌표 셀 주변으로 표면을 확장할 수 있다.

입력은 input_preview/model_input.png, 원본 크기 마스크는 bumper_mask_original.png,
제약 적용 전후 좌표 수는 console.log의 [Bumper support]로 확인한다.
결과는 bumper.glb이다. run.json의 complete는 실행 완료이며 범퍼만 생성됐다는
의미의 품질 승인은 아니다. 최종 외형, 뒷면 및 머티리얼 미리보기로 품질을 확인한다.

CPU 검증: 기존 ProjGrid와 투영 좌표 일치, 외곽 실루엣 처리, 빈 결과 거부,
희소 투영 및 기존 범퍼 테스트 13개 통과. GPU 추론/품질은 서버 재실행으로 검증해야 한다.

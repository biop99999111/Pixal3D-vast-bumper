# Linux GPU 서버: Dockerfile + Compose

현재 Vast.ai에서 진행 중인 추론은 그대로 둔다. 아래 구성은 Linux 호스트에서
새 이미지를 빌드하고 실행하기 위한 것이며, 실행 중인 컨테이너를 자동 변환하지 않는다.
Vast.ai 컨테이너 내부에서 Docker 엔진을 실행할 필요도 없다.

## 준비와 지원 범위

- Linux x86_64 호스트에 Docker Engine, Compose v2, 호환 NVIDIA 드라이버,
  NVIDIA Container Toolkit이 필요하다. `nvidia-smi`로 호스트 GPU를 확인한다.
- 기본 이미지는 `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`이다.
  Vast에서 확인한 Torch/CUDA 버전을 기준으로 하지만 배포판·Python까지 같은 이미지는 아니다.
- 기본 빌드 대상은 RTX 5090 (`12.0`)이다. RTX 4090은 `8.9`를 사용한다.
  두 GPU를 지원하려면 `.env`의 `CUDA_ARCH_LIST=8.9;12.0`과 이미지 태그를 함께 바꾼다.
  대상 GPU를 늘리면 컴파일 시간과 이미지 크기도 증가한다.
- GPU 없는 빌드 머신에서도 명시한 아키텍처로 컴파일하도록 구성했다.
  실제 GPU 연산 검사는 컨테이너 실행 후 수행한다.
- Blender는 이 이미지에 포함하지 않는다. 범퍼 추론·GLB export가 우선 범위이며,
  서버에서 Blender 렌더링까지 하려면 별도 검증된 Blender 설치 레이어를 추가해야 한다.

## 첫 실행

저장소를 clone한 Linux 호스트의 저장소 루트에서 실행한다.

```bash
mkdir -p inputs outputs cache
cp .env.example .env
id -u
id -g
```

`.env`의 `LOCAL_UID`, `LOCAL_GID`를 위 숫자로 맞춘다. 마운트할 폴더와
`notebooks`는 해당 계정이 쓸 수 있어야 한다. 컨테이너가 생성한 결과도 이 계정 소유가 된다.

```bash
docker compose config --quiet
docker compose build pixal3d
docker compose up -d
docker compose logs --tail=30 pixal3d
```

첫 빌드는 CUDA 확장을 컴파일하므로 오래 걸린다. 이후 같은 이미지 실행 시에는
설치 스크립트를 다시 실행하지 않는다. 로그의 Jupyter 토큰으로
`http://localhost:8888`에 접속한다. 원격 서버에서는 로컬 PC에서 SSH 터널을 연다.

```bash
ssh -L 8888:127.0.0.1:8888 USER@SERVER
```

Jupyter는 호스트의 loopback에만 포트를 공개하며 기본 토큰 인증을 유지한다.
노트북에서는 `Pixal3D GPU` 커널을 선택한다.

## 환경 검사와 모델 준비

```bash
docker compose exec pixal3d python scripts/check_environment.py --gpu-smoke
docker compose exec pixal3d hf auth login
docker compose exec pixal3d python scripts/prepare_models.py --background-mode auto
```

인증 토큰은 숨김 입력으로 전달한다. 모델 접근 권한은 해당 계정에 있어야 한다.
로그인은 실행 시 연결된 `cache/huggingface`에 보관되며 이미지에 넣지 않는다.
이 캐시 폴더에는 인증 정보도 있으므로 공개 저장소나 공개 이미지에 포함하지 않는다.

기존 Vast 모델 캐시를 옮기는 경우 Hugging Face의 `hub` 폴더는 호스트의
`cache/huggingface/hub`에, Torch Hub 캐시는 `cache/torch/hub`에 배치한다.
심볼릭 링크를 보존해 복사한다. 이전 서버의 절대 경로가 들어 있는 모델 lock은
직접 재사용하지 말고 위 `prepare_models.py`로 `/app` 경로의 lock을 생성한다.
이전 서버의 `/root/.cache` 전체를 공개 이미지에 복사하지 않는다.

## 범퍼 한 장 실행

```bash
docker compose exec pixal3d bash
```

컨테이너 터미널에서:

```bash
cp configs/vast/baseline.json outputs/bumper_config.json
sed -i 's/provided_mask/auto/' outputs/bumper_config.json
python -m bumper_synth infer \
  --image assets/images/bumper.jpg \
  --config outputs/bumper_config.json \
  --models-lock cache/models/models.lock.json \
  --output outputs/bumper_single_001
exit
```

호스트의 `outputs/bumper_single_001/bumper.glb`에서 결과를 확인한다.
`inputs`, `outputs`, `cache`, `notebooks`는 bind mount이므로 컨테이너를 교체해도 보존된다.
모델 가중치, 회사 입력, 결과는 이미지 빌드 대상에 포함하지 않는다.

## 유지보수와 버전 배포

- 코드만 수정하면 `docker compose build pixal3d`에서 의존성 컴파일 레이어를 재사용한다.
  requirements, 의존성 소스 ref, 설치 스크립트, GPU 빌드 옵션을 바꾸면 재빌드한다.
- 노트북은 호스트 파일을 연결하므로 수정 내용이 바로 보인다. 다른 애플리케이션 코드는
  이미지에 고정되므로 수정 후 rebuild와 `docker compose up -d`가 필요하다.
- `docker compose up -d`는 컨테이너를 교체할 수 있으므로 진행 중인 추론이 끝난 후 실행한다.
- 일반 Linux에서 빌드·추론까지 검증한 이미지에 릴리스 태그를 붙이고 레지스트리에 배포한다.
  다른 서버의 `.env`에서 `PIXAL_IMAGE`를 해당 태그나 digest로 지정한 뒤 실행한다.

```bash
docker compose pull pixal3d
docker compose up -d --no-build
```

이전 이미지로 복구할 때도 `.env`의 이미지 버전을 바꾸고 같은 명령을 사용한다.
노트북·설정도 이미지와 맞는 Git 릴리스 버전을 사용한다. 현재 저장소의 Compose 파일만으로
공개 레지스트리에 이미지가 게시되지는 않는다.

실제 이미지에 설치된 패키지 버전과 소스 ref는 `/opt/pixal3d-build/`에 보관한다.
PyTorch와 핵심 CUDA 확장 버전은 고정했지만 모든 전이 의존성을 완전히 잠근 것은 아니다.
릴리스에서는 검증된 이미지 digest를 배포하고, 필요하면 `BASE_IMAGE`도 digest로 고정한다.
Docker 빌드는 기본 이미지·빌드 캐시·최종 이미지 공간을 추가로 사용하므로
현재 Vast 서버의 모델 저장 공간 추산과 별도로 빌드 호스트 공간을 확보한다.

## 검증 상태

Compose 설정 파싱, 설치 스크립트 Bash 문법, Python 회귀 테스트를 로컬에서 확인한다.
작성 환경의 Docker 엔진이 실행 중이 아니므로 실제 이미지 빌드와 컨테이너 GPU 추론은 미검증이다.
기존 Vast 환경에서의 CUDA 검사 성공과 새 Docker 이미지 검증은 별개다.

참고: [Docker Compose GPU 설정](https://docs.docker.com/compose/how-tos/gpu-support/),
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

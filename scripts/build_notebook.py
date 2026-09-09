"""Generate the checked-in notebook with no outputs or credentials."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md("""# 범퍼 손상 합성데이터 — Vast.ai 작업 공간

GitHub 저장소를 Jupyter 터미널에서 clone한 뒤 저장소 루트에서 `bash scripts/setup_vast.sh`를 실행하세요.
커널을 **Pixal3D Vast**로 선택합니다. GPU 추론은 Linux Vast 서버에서, 렌더링은 설치된 Blender에서 실행합니다.
환경 준비만으로 범퍼 단독 생성·실제 손상 검출 성능이 보장되지는 않습니다. 각 단계의 미리보기를 검수하세요.
"""),
code("""from pathlib import Path
import sys, os, json, subprocess, getpass
from IPython.display import display, Image, FileLink, HTML
import ipywidgets as widgets
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'bumper_synth').is_dir())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from bumper_synth.common import read_json, write_json

def run_cmd(*args):
    process = subprocess.Popen([str(a) for a in args], cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    try:
        for line in process.stdout:
            token = os.environ.get('HF_TOKEN')
            print(line.replace(token, '[REDACTED]') if token else line, end='')
        if process.wait():
            raise RuntimeError('명령 실패: 위 로그와 outputs의 상태 파일을 확인하세요.')
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise

BLENDER = 'blender'  # 필요하면 실행 파일의 절대 경로로 변경
print('Repository:', ROOT)
"""),
md("## 1. 환경 진단"),
code("run_cmd(sys.executable, 'scripts/check_environment.py', '--gpu-smoke')"),
md("## 2. Hugging Face 인증 — 필요한 경우만 실행\n토큰은 숨김 입력으로 받아 현재 커널 환경변수에만 저장합니다. 모델 사용 조건 동의가 필요한 경우 해당 모델 페이지에서 먼저 완료하세요."),
code("""token = getpass.getpass('HF token (공개 다운로드만 사용하면 Enter): ')
if token:
    os.environ['HF_TOKEN'] = token
del token
"""),
md("## 3. 입력 이미지 업로드\n권장: 범퍼만 분리된 투명 PNG. RGB 이미지만 있으면 다음 설정에서 `auto`를 선택하세요. 업로드하지 않으면 포함된 `assets/images/bumper.jpg`를 사용합니다. 업로드 파일은 Git 제외 디렉터리에 저장됩니다."),
code("""uploader = widgets.FileUpload(accept='.png,.jpg,.jpeg', multiple=False)
display(uploader)
"""),
code("""if uploader.value:
    item = uploader.value[0]
    input_path = ROOT/'inputs'/Path(item['name']).name
    input_path.parent.mkdir(exist_ok=True)
    input_path.write_bytes(bytes(item['content']))
else:
    input_path = ROOT/'assets/images/bumper.jpg'
display(Image(filename=str(input_path), width=500))
mask_path = None  # 외부 마스크 사용 시 Path('inputs/mask.png')
"""),
md("## 4. 생성 설정\n표준 1536 기준입니다. 동일 해상도에서 메모리 모드를 비교하세요. 범퍼만 생성하도록 강제하는 모델 옵션은 없습니다."),
code("""background = widgets.Dropdown(options=['provided_mask','auto'], value='auto', description='배경')
resolution = widgets.Dropdown(options=[1024,1536], value=1536, description='생성 해상도')
low_vram = widgets.Checkbox(value=False, description='Low VRAM')
steps = widgets.IntSlider(value=12,min=4,max=36,description='Steps')
texture_size = widgets.Dropdown(options=[2048,4096],value=4096,description='텍스처')
display(widgets.VBox([background,resolution,low_vram,steps,texture_size]))
"""),
code("""cfg = read_json('configs/vast/baseline.json')
cfg.update(background_mode=background.value,resolution=resolution.value,low_vram=low_vram.value)
for stage in cfg['samplers'].values(): stage['steps'] = steps.value
cfg['export']['texture_size'] = texture_size.value
write_json('outputs/current_config.json',cfg)
run_dir = ROOT/'outputs'/'bumper_001'  # 입력/설정을 바꾸면 새 이름 사용
run_cmd(sys.executable,'scripts/prepare_models.py','--background-mode',background.value)
models_lock = ROOT/'cache/models/models.lock.json'
"""),
md("## 5. 범퍼 생성\n실패 시 run.json과 stages.json에 실패 상태를 남깁니다. 설정이 같은 완료 실행은 재사용합니다."),
code("""args = [sys.executable,'-m','bumper_synth','infer','--config','outputs/current_config.json',
        '--image',input_path,'--output',run_dir,'--models-lock',models_lock]
if mask_path: args += ['--mask',mask_path]
run_cmd(*args)
display(Image(filename=str(run_dir/'model_input.png'),width=500))
display(FileLink(str((run_dir/'bumper.glb').relative_to(ROOT))))
print(read_json(run_dir/'quality.json'))
"""),
md("## 6. GLB 회전 확인과 입력 정합성\nGLB는 Blender에서 회전해 범퍼 단독 형상, 구멍, 깨끗한 표면을 확인하세요. 아래 비교는 저장된 입력 카메라와 export 좌표 변환을 적용합니다."),
code("""run_cmd(sys.executable,'-m','bumper_synth','preview','--asset',run_dir/'bumper.glb',
        '--output',run_dir/'preview','--blender',BLENDER)
display(Image(filename=str(run_dir/'preview/turntable.png'),width=700))
run_cmd(sys.executable,'-m','bumper_synth','align','--run',run_dir,'--blender',BLENDER)
display(Image(filename=str(run_dir/'alignment/overlay.png'),width=500))
display(Image(filename=str(run_dir/'alignment/difference.png'),width=500))
print(read_json(run_dir/'alignment/metrics.json'))
"""),
md("## 7. 부품 검수 기록\n차체·바퀴가 붙어 있거나 원본에 없는 가짜 손상이 있으면 체크하지 마세요. 기본 부품이 합격해야 손상 데이터로 진행합니다."),
code("""accepted = widgets.Checkbox(value=False,description='범퍼 단독 형상·깨끗한 표면·텍스처 검수 완료')
note = widgets.Textarea(placeholder='검수한 내용과 사용할 시점 범위를 기록하세요.')
display(accepted,note)
"""),
code("""if not accepted.value or not note.value.strip():
    raise ValueError('부품을 검수한 후 체크하고 내용을 입력하세요.')
run_cmd(sys.executable,'-m','bumper_synth','review','--asset',run_dir/'bumper.glb','--note',note.value)
"""),
md("## 8. 손상·RGB·YOLO 라벨 생성\n초기에는 소량 생성 후 미리보기를 확인하세요. `dent/딩(ding)/scratch`는 0/1/2입니다. 찍힘의 업무 정의는 실제 사진과 맞춰야 합니다."),
code("""render_cfg = read_json('configs/vast/render_high_quality.json')
write_json('outputs/current_render.json',render_cfg)
dataset_dir = ROOT/'outputs'/'dataset_bumper_001'
run_cmd(sys.executable,'-m','bumper_synth','render','--asset',run_dir/'bumper.glb',
        '--config','outputs/current_render.json','--output',dataset_dir,'--blender',BLENDER,
        '--group','part_001','--split','train')
for path in sorted((dataset_dir/'previews').glob('*.png'))[:12]:
    display(Image(filename=str(path),width=380))
"""),
md("## 9. 데이터 내보내기\n다른 원본 범퍼를 val/test 그룹으로 생성한 후 `assemble` 명령으로 data.yaml을 만드세요. 한 범퍼의 각도만 나눠 train/val로 사용하면 안 됩니다. 학습은 별도 환경의 scripts/train_yolo.py를 사용합니다."),
code("""import zipfile
archive = ROOT/'outputs'/'dataset_bumper_001.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['images','labels','masks','previews','metadata']:
        for path in (dataset_dir/folder).rglob('*'):
            if path.is_file(): z.write(path,path.relative_to(dataset_dir))
    z.write(dataset_dir/'dataset.json','dataset.json')
display(FileLink(str(archive.relative_to(ROOT))))
"""),
md("""## 선택 기능

- `python scripts/run_experiments.py --help`: 동일 입력에서 표준/low VRAM·해상도·steps 비교.
- `python -m bumper_synth export --help`: 저장한 mesh_state.npz에서 출력 설정만 바꿔 GLB 재출력.
- `python -m bumper_synth retexture --help`: 깨끗한 기본 부품의 정면 SD inpaint와 UV 베이킹. 실험 기능이며 별도의 inpainting 모델이 필요합니다. 라벨이 생성된 손상 이미지에는 적용하지 않습니다. 결과 GLB는 다시 검수해야 합니다.
- `python -m bumper_synth fixture --help`: 모델 다운로드 없이 Blender 설치와 라벨 생성을 확인할 시험 패널. 학습데이터로 사용하지 않습니다.

전체 안내: `docs/VAST_QUICKSTART.ko.md`. 노트북을 GitHub에 올리기 전 출력과 업로드 위젯 상태를 제거하세요.
"""),
]
nb.metadata = {"kernelspec":{"display_name":"Pixal3D Vast","language":"python","name":"pixal3d-vast"},
               "language_info":{"name":"python"}}
for index, cell in enumerate(nb.cells):
    cell["id"] = f"workspace-{index:02d}"
target=ROOT/'notebooks/workspace.ipynb'
target.parent.mkdir(exist_ok=True)
nbf.write(nb,target)
print(target)

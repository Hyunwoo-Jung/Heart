# MC Q baseline

`DH_test`와 독립적인 신경망 Monte Carlo control 기준선이다. `heart/`의 환경과
medium 규칙 상대 3명만 재사용한다. 공식 제출용 JAX export는 포함하지 않는다.

## 학습 정의

- 환경: `classic-v0`, 100점 종료까지 전체 매치. 패스와 플레이 모두 학습.
- 목표: 자신의 매치 누적 벌점 최소화. 우승 확률을 직접 최적화하지 않는다.
- 데이터: 현재 ε-greedy 정책을 고정하여 병렬 매치를 끝까지 수집한다.
- 매치 묶음마다 학습 좌석을 `0 → 1 → 2 → 3`으로 순환한다.
- target: `G_t = sum(rewards[t:])`, `gamma=1`. 딜 경계에서도 누적을 계속한다.
- 수치 스케일만 줄이기 위해 `G_t / 100`에 Q를 회귀한다.
- 손실: 선택 행동의 Q와 target 사이 MSE. 새 데이터에 한 번의 shuffled
  minibatch epoch를 적용한 뒤 폐기한다. TD bootstrap, replay, CQL, 순위 보조
  손실, self-play는 사용하지 않는다.
- 탐험: 합법 행동 안에서 ε-greedy, 기본 ε=0.1. 평가 시에는 greedy.
- 모델: 256-unit ReLU 은닉층 2개인 MLP, 패스 286개 + 카드 ID 52개 Q 출력.
  학습 환경의 플레이 입력은 13개 슬롯이므로 전달 직전에 카드 ID를 변환한다.
- 입력: 자기 손패, 순서를 보존한 공개 트릭 이력, 트릭 승자, 현재 트릭,
  자신이 패스한/받은 카드, 공개 벌점·매치 점수·손패 크기와 진행 정보.
  정책 입력에는 상대 손패가 포함되지 않는다.

신경망 근사와 불완전 정보 때문에 표 기반 MC control의 수렴을 보장하지 않는다.
고정 medium 상대에서의 성능과 다른 학습 정책을 상대할 때의 성능도 다를 수 있다.

## 설치와 GPU 실행

저장소 루트에서 실행한다. GPU에 맞는 PyTorch 빌드가 설치되어 있어야 한다.
`pip install -e '.[mc]'`는 PyTorch를 포함한 선택 의존성을 설치한다.
CUDA 드라이버와 PyTorch의 GPU 호환 여부는 실행할 머신에서 확인한다.

```bash
python -m pip install -e '.[mc]'
bash baselines/mc/train.sh
```

실행 설정은 **`train.sh` 안에서 수정**한다. 기본값은 다음과 같다.

```bash
export CUDA_VISIBLE_DEVICES=2
CONDA_ENV="openal"
SEEDS=(1)
ITERATIONS=1000
NUM_ENVS=16
BATCH_SIZE=256
HIDDEN=256
LR=0.0003
EPSILON=0.1
EVAL_SEED=1000000
EVAL_EVERY=25
RESUME=""
```

스크립트가 `~/anaconda3/etc/profile.d/conda.sh`를 읽고 `openal`을 활성화한다.
다른 conda 설치 경로라면 `CONDA_SH`를 수정한다. 여러 seed를 실행하려면
`SEEDS=(1 2 3)`으로 바꾼다. 각 seed는 새 모델과 출력 폴더로 **순차 실행**되며,
한 작업이 실패하면 종료 코드를 보존하고 다음 작업을 시작하지 않는다.

기본값은 1,000회 수집 × 16매치 = **16,000개 학습 매치**이다. 평가 매치는
별도다. `ITERATIONS`는 optimizer step 수가 아니라 매치 수집·학습 묶음 수다.
`NUM_ENVS`는 기본 평가에서도 좌석별 매치 수로 쓰인다. 기본 평가는 총 64매치.
0번째 모델과 매 25회, 마지막 반복에서 평가한다. 첫 JAX 컴파일에는 시간이 든다.

스크립트는 **JAX 게임 환경을 CPU**, **PyTorch 모델을 선택한 GPU**에 둔다.
따라서 JAX CUDA 설치가 필요 없고 JAX의 GPU 메모리 선점도 피한다.
물리 GPU 2번은 프로세스 내부에서는 `cuda:0`으로 보인다.
Python 모듈을 직접 실행할 때도 `JAX_PLATFORMS=cpu`를 지정하는 것을 권장한다.

작은 동작 확인만 하려면 스크립트에서 `ITERATIONS=1`, `NUM_ENVS=2`,
`HIDDEN=32`로 변경한다. 이는 성능 판단용 학습량이 아니다.

작업 이름은 `Heart_<시간>_mc_medium_seed<seed>` 형태다.

- 체크포인트와 지표: `.local_training/mc/<JOB_NAME>/`
- 표준 출력: `logs/<MMDD>/<JOB_NAME>.out`
- 표준 오류·실행 명령·PID·완료/실패 상태: `logs/<MMDD>/<JOB_NAME>.log`

학습 프로세스는 `nohup`으로 시작하며, 스크립트는 `wait`로 완료를 기다린다.
출력은 `tee`로 터미널과 파일에 함께 기록한다. 전체 seed 목록을 터미널 종료와
무관하게 실행하려면 런처 자체를 `nohup bash baselines/mc/train.sh > launcher.out 2>&1 &`
형태로 시작한다.

## 저장, 평가와 이어하기

- `last.pt`: 매 학습 반복 종료 후 모델·optimizer·학습 RNG·설정·반복 수 저장.
- `best.pt`: 별도 고정 시드, 네 좌석 평가의 **평균 매치 벌점**이 가장 낮은 모델.
- `metrics.jsonl`: 설정, loss, 수집 시간, 벌점, 평균 순위, 공동 우승률.
- 평균 순위는 1부터 시작하고 동점은 평균 순위, 공동 최저점은 모두 우승이다.
- `penalty_per_deal`도 함께 기록하지만 모델 선택은 `mean_penalty` 기준이다.

이어하려면 `train.sh`의 `RESUME`에 실제 `last.pt` 경로를 넣고,
`SEEDS`는 해당 체크포인트의 seed 하나로 지정한다. 예를 들어:

```bash
SEEDS=(1)
ITERATIONS=2000
RESUME=".local_training/mc/<기존 JOB_NAME>/last.pt"
```

수정 후 같은 `bash baselines/mc/train.sh` 명령으로 실행한다.
`ITERATIONS=2000`은 재개 후 총 반복 수다. seed, 평가 seed, 모델 크기,
학습률, ε, batch 크기, 환경 수는 저장 시 설정과 같아야 한다. 기본값을 바꿔
학습했다면 재개할 때 같은 설정을 사용한다. 체크포인트는 기존 폴더에 이어 쓰고
터미널 로그는 새 작업 이름으로 생성한다. 실행 중단 시 마지막 완료 반복에서 재개한다.
하드웨어·라이브러리 버전이 바뀌면 비트 단위 재현은 보장하지 않는다.

독립 seed 비교는 **`RESUME=""`로** 실행한다.
학습 중 모델 선택용 기본 평가 seed는 1,000,000이며 아래 최종 대국 평가의
기본 seed 2,000,000과 분리되어 있다. 다른 상대·충분한 매치 수·여러 학습 seed로
최종 성능을 비교해야 한다.

## 다른 방법론과 대국

체크포인트는 공통 `SeatPolicy(state, player, key) -> action`으로 불러온다.
카드 플레이는 카드 ID, 패스는 조합 ID를 반환한다. 내부적으로는 개인 관측만
인코딩한다. 로더의 기본 추론 장치는 CPU다.

```python
from baselines.mc.policy import make_policy

policy = make_policy(".local_training/mc/<JOB_NAME>/best.pt", device="cuda")
action = policy(state, player, key)
```

동일 seed의 초기 배분을 네 학습자 좌석에서 반복하여 평가한다. `--games`는
4의 배수여야 한다. 상대는 남은 좌석의 번호 순서로 배치된다.

```bash
JAX_PLATFORMS=cpu python -m baselines.mc.evaluate \
  ".local_training/mc/<JOB_NAME>/best.pt" \
  --opponents medium medium medium --games 100

JAX_PLATFORMS=cpu python -m baselines.mc.evaluate \
  ".local_training/mc/<JOB_NAME>/best.pt" \
  --opponents hard hard hard --games 100
```

다른 방법론도 `heart.play.load_seat_policy`가 읽는 factory 규약을 구현하면 된다.
예를 들어 `my_method.policy.make_policy(checkpoint_path)`가 같은 인터페이스의
callable을 반환하면 다음처럼 대국할 수 있다.

```bash
JAX_PLATFORMS=cpu python -m baselines.mc.evaluate \
  ".local_training/mc/<JOB_NAME>/best.pt" \
  --opponents \
  'my_method.policy:make_policy(path/to/model.pt)' medium medium \
  --games 100
```

사람과 대국하는 기존 UI에도 연결할 수 있다.

```bash
JAX_PLATFORMS=cpu python -m heart \
  --seat 'baselines.mc.policy:make_policy(.local_training/mc/<JOB_NAME>/best.pt)'
```

## 작은 검증

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p 'test_mc*.py' -v
```

누적 return, 동점 지표, 공개 이력 인코딩, 마스크와 카드/슬롯 변환, 완주한
병렬 매치의 보상 합계, 학습 업데이트, 체크포인트 복원과 정책 로딩을 검사한다.
런처 검증은 모의 conda/Python을 사용해 seed별 순차 실행, 로그, 실패 코드,
재개 경로를 검사하며 실제 GPU 학습은 실행하지 않는다.

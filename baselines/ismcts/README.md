# ISMCTS baseline

`DH_test` 및 MC 신경망과 독립적인 SO-ISMCTS 기준선이다. 사전 학습이나
체크포인트 없이 매 의사결정에서 새 트리를 만든다. HEART의 게임 규칙과
medium 규칙 정책을 재사용한다.

## 설정과 알고리즘

| 항목 | 구현 |
| --- | --- |
| 관측 | 자기 손패, 공개 트릭 이력·승자·벌점, 현재 트릭, 손패 크기, 자신의 패스/수신 카드 |
| 숨은 손패 | 관측과 합치하는 배분에서 매 시뮬레이션 새로 샘플링 |
| 행동 | 패스 3장 조합 286개, 플레이는 합법 카드 ID |
| 탐색 범위 | 현재 딜 종료까지, 다음 딜을 미리 생성하지 않음 |
| 보상 | 딜 종료 후 각 플레이어의 `-scores / 26`, 슈팅 더 문 정산 포함 |
| 선택 | 행동 가용 횟수를 반영한 UCB, `C=sqrt(2)`, 해당 차례 플레이어의 보상 사용 |
| 확장 이후 | 네 플레이어 모두 medium으로 딜 끝까지 rollout |
| 최종 행동 | 루트 방문 횟수 최대, 동률이면 자신의 평균 보상 최대 |
| 예산 | 결정당 정확히 1,024회, 배치 32개, 진행 중 경로에 virtual loss 적용 |
| 실제 평가 | medium 3명과 100점 종료까지 전체 매치, 기본 96매치 |

이미 공개된 카드, suit void, 첫 트릭의 벌점 카드 제한, 하트 브레이크 전
하트 리드 제한과 자신이 전달한 카드의 수신자를 샘플링에 반영한다.
조건에 맞는 배분이 없으면 오류를 내며 제약을 임의로 완화하지 않는다.
이전 상대 행동의 확률까지 추론하는 posterior는 사용하지 않는다.

패스 탐색에서는 상대 3명의 패스를 샘플 손패에서 medium으로 다시 계산하고
동시에 교환한다. 실제 게임에 저장된 상대 패스 선택이나 미래 딜 RNG는
검색에 전달하지 않는다. 패스 이후 트리 분기는 자신이 받은 카드와 공개된
선공 좌석만 사용한다. 플레이 트리는 관찰한 공개 행동 이력으로 공유한다.

## 실행

저장소 루트에서 실행한다.

```bash
bash baselines/ismcts/run.sh
```

설정은 `run.sh` 상단에서 수정한다.

```bash
export CUDA_VISIBLE_DEVICES=2
CONDA_ENV="heart"
SEEDS=(2000000)
ITERATIONS=1024
BATCH_SIZE=32
EXPLORATION=1.4142135623730951
GAMES=96
OPPONENTS=("medium" "medium" "medium")
```

`ITERATIONS`는 학습 반복 수가 아니라 **매 결정의 탐색 횟수**다.
seed는 초기 배분과 탐색 난수를 제어한다. 여러 seed는 순차 실행된다.
동일 라이브러리·하드웨어·배치 크기를 유지하여 비교한다.

트리와 손패 샘플링, 실제 매치 진행은 CPU에서 처리하고, 탐색 상태 전이와
rollout은 JAX `jit/vmap`으로 선택한 GPU에서 처리한다. GPU가 없으면
오류로 종료하며 자동 CPU 대체는 하지 않는다. PyTorch의 CUDA 지원만으로는
충분하지 않다. `heart` 환경에 드라이버와 호환되는 JAX CUDA 빌드가 필요하다.
설치는 [JAX 공식 안내](https://docs.jax.dev/en/latest/installation.html)를 따른다.

로그는 `logs/<MMDD>/Heart_<시간>_ismcts_iter1024_seed<seed>.out/.log`에
기록한다. `.out`에는 설정, 매치별 점수·좌석·판단 시간, 최종 요약이 들어간다.
요약은 평균 매치 벌점, 딜당 벌점, 평균 순위, 공동 우승률과 판단 시간
평균/p95를 포함한다. **첫 JIT 컴파일 시간도 판단 시간에 포함**된다.
매치가 끝나기 전에는 시작 설정만 출력될 수 있다.

런처는 `nohup` 자식 프로세스를 `wait`하며 실패 시 다음 seed를 중단한다.
터미널 종료 후에도 전체 seed 목록을 계속 실행하려면:

```bash
nohup bash baselines/ismcts/run.sh > launcher.out 2>&1 &
```

## 다른 방법론과 대국

공통 `SeatPolicy(state, player, key) -> action` 인터페이스를 사용한다.
검색으로는 이 상태에서 추출한 개인 관측만 전달된다.

```python
from baselines.ismcts.policy import make_policy

policy = make_policy(1024, batch_size=32, device="gpu")
action = policy(state, player, key)
```

MC와 medium 2명을 상대로 비교하는 예:

```bash
CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m baselines.ismcts.evaluate \
  --opponents 'baselines.mc.policy:make_policy(path/to/best.pt)' medium medium \
  --games 96 --seed 2000000
```

MC를 상대에 넣을 때만 PyTorch가 필요하다. factory로 불러오는 MC 추론은
기본 CPU다. ISMCTS factory 문자열은
`baselines.ismcts.policy:make_policy(1024)`이며 기본 GPU를 사용한다.

공통 평가기는 같은 초기 배분을 모든 서로 다른 좌석 배치에서 반복한다.
상대가 모두 같으면 4개, 다른 상대 하나와 같은 상대 둘이면 12개, 모두
다르면 24개 배치다. `GAMES`는 해당 수의 배수여야 한다.

## 해석상 주의

- MC는 매치 전체의 return을 학습하지만 ISMCTS는 현재 딜 벌점만 최적화한다.
  최종 평가는 같은 전체 매치 지표로 비교하되 목적 범위 차이를 명시한다.
- 패스 후보가 286개이므로 1,024회는 후보당 평균 약 3.6회에 불과하다.
  이는 초기 기준선 예산이며 충분히 수렴한 탐색으로 해석하지 않는다.
- 상대 노드도 자기 벌점을 줄이는 행동을 탐색한다. medium은 실제 기본 상대,
  숨은 패스 모델 및 rollout 정책이다. 트리 전체를 medium으로 고정하지 않는다.
- SO-ISMCTS는 상대별 지식 상태를 완전히 분리하지 않는다. 상대의 실제 전략을
  학습하지 않으며 다른 방법론과 대국해도 숨은 패스 모델은 medium이다.
- CPU 샘플링과 GPU 통신 비용이 있어 GPU가 항상 더 빠르지는 않다.
  배치 크기는 탐색 순서에도 영향을 주므로 실험 설정으로 함께 기록한다.

## 검증

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p 'test_ismcts.py' -v
JAX_PLATFORMS=cpu python -m baselines.ismcts.evaluate \
  --device cpu --iterations 4 --batch-size 4 --games 4
```

테스트는 샘플 손패의 공개 이력 합법 재생, 패스 교환, 숨은 상태 비참조,
정확한 예산, 슈팅 더 문 보상, 좌석 균형을 검사한다.
두 번째 명령은 전체 매치 연결 확인용이며 성능 평가용 예산이 아니다.

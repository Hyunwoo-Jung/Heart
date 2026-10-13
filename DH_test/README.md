# DH_test: Heart 강화학습 학습기

이 문서는 `DH_test/train.py`의 학습 방식과 보상 정의, 주요 하이퍼파라미터, 실행 방법을 설명합니다. 구현은 Mortal의 로그 기반 학습에서 쓰는 Q 회귀, CQL 정규화, 순위 보조 예측 구조를 Heart의 `classic-v0`에 맞게 옮긴 실험용 스캐폴드입니다.

## 학습 개요

Heart 저장소에는 Mortal처럼 대규모 경기 로그를 읽는 데이터 파이프라인이 없습니다. 따라서 이 학습기는 hard 규칙 정책과 현재 모델을 섞어 `classic-v0` 자가 대국을 만들고, 그 경험을 replay buffer에서 샘플링합니다. 4개 좌석 모두 기록 대상이며, 모델 행동 외의 규칙 행동도 학습 데이터에 포함됩니다.

학습은 행동 가치 함수 `Q(s, a)`를 회귀하는 Monte Carlo 방식입니다. 각 상태에서 선택한 행동의 Q 값이 그 행동 이후 매치 종료까지 받은 누적 보상에 가까워지도록 학습합니다. Heart 보상이 deal 종료에만 주어지므로 엔진 이벤트마다 중간 보상은 0이고, 할인율은 `gamma = 1`로 둡니다.

이 구현은 Mortal의 실제 로그 학습과 같지는 않습니다. CQL을 쓰는 로그 기반 오프라인 학습 대신, 규칙 정책과 현재 모델이 생성한 데이터를 계속 추가하는 혼합 정책 자가 대국을 씁니다.

## 보상과 학습 목표

Heart의 deal 종료 보상은 각 플레이어의 moon 조정 penalty score에 음수를 붙인 값입니다.

```text
r_i = -effective_deal_score_i
G_t = sum of r_i from the chosen action through match termination
```

따라서 점수가 낮을수록 좋은 Hearts의 목표와 보상의 부호가 일치합니다. Moon shot이 발생하면 shooter는 해당 deal에서 `0`, 나머지 플레이어는 각각 `-26`을 받습니다. 이 학습기는 deal 경계 보상을 이후 매치 종료까지 더해 해당 플레이어의 행동마다 Monte Carlo target `G_t`를 만듭니다. discounting은 적용하지 않습니다.

전체 손실은 다음과 같습니다.

```text
L = 0.5 * MSE(Q(s, a), G_t)
  + 0.1 * CQL(s, a)
  + 0.2 * CrossEntropy(rank_logits, final_rank)

CQL(s, a) = logsumexp(Q(s, legal_actions)) - Q(s, a)
```

- **Q 회귀:** 실제 선택 행동의 Q 값을 매치 종료까지의 누적 보상에 맞춥니다.
- **CQL 정규화:** 해당 관측에서 legal mask가 허용한 행동들의 Q 값이 선택 행동보다 과도하게 커지지 않도록 제약합니다. 불법 행동은 `logsumexp`에서 제외됩니다.
- **순위 보조 목표:** 같은 상태 표현에서 매치 최종 순위를 4개 등급으로 분류합니다. 0등이 가장 낮은 penalty score이며, 동점은 현재 구현의 안정 정렬에 따라 좌석 순서로 등급이 나뉩니다.

## 상태와 행동 표현

모델 입력은 52장의 카드 위치를 길이 축으로 둔 `(29, 52)` 텐서입니다. 채널에는 현재 플레이어 손패, 공개된 플레이 이력과 현재 트릭, 본인이 패스한/받은 카드, 매치 점수, 각 손패 크기, 플레이어·리더, 트릭 진행, 하트 브레이크 여부, phase와 패스 방향이 들어갑니다. 상대의 비공개 손패는 입력하지 않습니다.

네트워크는 Mortal과 비슷한 residual Conv1D 인코더를 공유하고, Q 행동 공간을 하나의 338개 출력으로 구성합니다.

- `0..285`: 13장 중 3장을 고르는 패스 조합 286개
- `286..337`: 카드 플레이 52개

현재 phase의 action mask만 활성화됩니다. 추론 때는 활성 행동 중 Q 값이 가장 높은 행동을 선택합니다.

## 데이터 수집과 기본 설정

기본값은 `train.py`의 argparse 설정입니다.

| 설정 | 기본값 | 의미 |
| --- | ---: | --- |
| `--rule-mix` | `0.75` | 각 행동에서 hard 규칙 정책을 고를 확률 |
| `--epsilon` | `0.05` | 규칙 정책을 고르지 않은 경우 모델 행동 대신 legal 행동을 무작위 선택할 확률 |
| `--warmup-matches` | `100` | 학습 업데이트 전에 수집할 최소 매치 수 |
| `--replay-capacity` | `250000` | 저장할 최대 상태-행동 사례 수 |
| `--batch-size` | `256` | 업데이트마다 replay에서 샘플링할 사례 수 |
| `--updates-per-match` | `100` | 새 매치 하나 수집 후 수행할 최대 업데이트 수 |
| `--lr` | `1e-4` | AdamW 초기 학습률 |
| `--cql-weight` | `0.1` | CQL 손실 가중치 |
| `--rank-weight` | `0.2` | 최종 순위 분류 손실 가중치 |
| `--steps` | `100000` | 총 optimizer 업데이트 수 목표 |

가중치는 Mortal의 원래 설정을 그대로 복사한 값이 아니라 Heart의 보상 스케일에 맞춘 초기 실험값입니다. 학습 결과에 따라 조정해야 합니다.

## 평가와 체크포인트

`--eval-every` 업데이트마다 고정된 평가 시드로 평가 매치를 실행합니다. 학습 네트워크는 좌석 0에서 hard 규칙 정책 3개와 대국합니다. 평균 순위, 평균 penalty score, 1위율, 4위율을 출력하며, 매번 같은 평가 시드와 대국 구성을 사용해 기존 최고 성능과 비교합니다. 기본 평가 매치 수는 `--eval-games 20`입니다.

체크포인트는 `DH_test/checkpoints/best.pt` 하나만 사용합니다. 학습 시작 시 이 파일이 있으면 모델과 AdamW 상태, 누적 업데이트 수, 최고 평가 순위를 자동으로 불러옵니다. 파일이 없으면 초기 모델을 평가해 기준 체크포인트를 만든 뒤 학습을 시작합니다. 이후 평가 평균 순위가 기존 최고보다 낮을 때만 `best.pt`를 덮어씁니다. 따라서 평가에서 개선되지 않은 마지막 가중치는 저장되지 않습니다. replay buffer는 체크포인트에 포함하지 않아 실행을 재개할 때 새 매치 데이터부터 다시 모읍니다.

## 훈련 파일 사용 방법

명령은 Heart 프로젝트 루트에서 실행합니다. PyTorch는 설치 환경에 맞는 CPU 또는 CUDA 빌드를 설치하세요.

```bash
python -m pip install -e .
python -m pip install torch
python -m DH_test.train --steps 100000 --device cuda
```

여러 무작위 seed로 학습을 반복할 때는 실행 가능한 `train_repeats.sh`를 사용합니다. GPU 번호는 PyTorch의 `cuda:N` 장치 번호이며, 반복은 순차 실행됩니다. 모든 반복은 동일한 `DH_test/checkpoints/best.pt`에서 시작합니다. 앞선 반복이 평가 최고 기록을 갱신했다면 다음 반복이 그 체크포인트를 불러오므로, 여러 seed로 차례로 이어서 개선할 수 있습니다.

```bash
./DH_test/train_repeats.sh \
  --gpu 1 \
  --repeats 5 \
  -- \
  --steps 100000 \
  --eval-games 20
```

위 명령은 GPU 1에서 무작위 seed를 바꿔 5회 순차 학습하고, 최고 체크포인트만 `DH_test/checkpoints/best.pt`에 유지합니다. CPU에서 반복하려면 `--gpu cpu`를 지정합니다. 스크립트가 `--device`, `--seed`를 각 실행마다 지정합니다. 체크포인트 경로는 `train.py`에서 `DH_test/checkpoints`로 고정되어 있습니다.

CPU로 짧게 실행하는 예시는 다음과 같습니다.

```bash
python -m DH_test.train \
  --steps 5000 \
  --warmup-matches 20 \
  --updates-per-match 10 \
  --eval-every 1000 \
  --eval-games 4 \
  --device cpu
```

단일 학습 실행도 기본으로 같은 최고 체크포인트를 자동 복원합니다.

```bash
python -m DH_test.train \
  --steps 100000 \
  --device cuda
```

`--steps`는 이번 실행에서 수행할 업데이트 수입니다. 체크포인트의 누적 업데이트 수와 별도로 매 실행마다 해당 횟수만큼 추가 학습합니다.

주요 수집/학습 설정은 명령행에서 바꿀 수 있습니다.

```bash
python -m DH_test.train \
  --rule-mix 0.6 \
  --epsilon 0.05 \
  --batch-size 256 \
  --cql-weight 0.1 \
  --rank-weight 0.2 \
  --eval-every 1000
```

## 학습 정책 사용

저장된 `best.pt`는 Heart의 seat 정책 팩토리로 불러올 수 있습니다.

```bash
python -m heart --seat 'DH_test.policy:make_policy(DH_test/checkpoints/best.pt)'
```

기본 정책은 greedy Q 선택이며, 별도 device를 지정하려면 팩토리 인자로 전달합니다.

```bash
python -m heart --seat 'DH_test.policy:make_policy(DH_test/checkpoints/best.pt,cuda)'
```

## 범위와 한계

이 구현은 Mortal의 분산 수집기, 대규모 기보 데이터셋, TensorBoard 로그를 포함하지 않습니다. 자가 대국과 단일 프로세스 replay buffer를 사용하며, 기본 모델도 연구 실험용 크기입니다. 따라서 현재 하이퍼파라미터와 구조는 재현된 Mortal 성능을 보장하지 않고, 실제 강도는 학습 및 평가 결과로 확인해야 합니다.

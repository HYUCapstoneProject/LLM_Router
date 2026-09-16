# ML 개발 질문·고민·의사결정 기록

상태: **설계 근거 기록**

기준일: 2026-08-14

## 1. 문서 목적

이 문서는 ML 부분을 개발하면서 제기한 질문, 확인한 사실, 선택한 방향과 아직 남은 고민을
한 곳에 기록한다. 단순한 작업 일지가 아니라 “왜 현재 구조가 되었는가”를 나중에 설명하기
위한 의사결정 지도다.

- 현행 작업 상태와 체크박스: `docs/ml/workstream-checklist.md`
- 팀에서 확정할 계약: `docs/ml/backend-infrastructure-agreement.md`
- 세부 알고리즘과 artifact 계약: RouterBench, RouterArena, ModernBERT 각 명세

상태 표기는 `결정`, `확인`, `실험 필요`, `팀 협의`로 통일한다.

## 2. RouterBench 데이터와 난이도 라벨

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| RouterBench의 규모와 형식은 무엇인가 | 공식 Pickle의 canonical 질문 36,497개 중 초기 핵심 범위 34,778행을 라벨링했다. 감사·학습용 Parquet/JSON으로 변환한다. | 확인 |
| 난이도를 무엇으로 정의할 것인가 | 교과 난이도가 아니라 질문을 만족스럽게 해결할 수 있는 최소 모델 capability tier로 정의한다. | 결정 |
| 대표 모델 3개만 사용하면 충분한가 | 특정 모델의 우연한 강·약점에 민감하다. 11개 모델을 empirical low/medium/high tier로 나누고 tier consensus를 사용하면 모델군 수준의 해결 가능성을 반영해 일반화에 유리하다. | 결정 |
| solve rate는 주 라벨 기준인가 | v2까지는 tier consensus를 주 라벨로 사용했다. 2026-09-02 v3 실험은 전체 11개 모델의 solve rate를 주 라벨로 사용하며, RouterBench 내부 probe에서 후보 E/F/G 중 kappa가 가장 높은 G(`0.64/0.27`)를 선택한다. 의미가 최소 충분 tier에서 내재 난이도로 바뀌므로 Claude tier 표본 검증 전에는 production 계약을 대체하지 않는다. | 실험 필요 |
| 라벨링 예시는 어떻게 해석하는가 | low tier가 안정적으로 성공하면 easy, low는 부족하고 medium이 성공하면 medium, high가 필요하거나 전체가 실패하면 hard다. | 결정 |
| 100행 smoke로 라벨 품질을 확정할 수 있는가 | smoke는 schema, 다운로드, 라벨링, 저장 경로의 동작 확인용이다. threshold 품질 판단은 전체 분포와 실제 Claude 표본 검증이 필요하다. | 확인 |
| RouterBench 원본을 그대로 학습에 넣어도 되는가 | few-shot wrapper에서 실제 query를 분리하고, 중복 그룹으로 split 누수를 막고, 정답·모델 성능 컬럼을 학습 입력에서 제거해야 한다. | 결정 |
| 전체 데이터 처리 결과는 무엇인가 | 핵심 34,778행 라벨은 easy 12,765, medium 5,348, hard 16,665이고 학습 projection은 34,764행이다. | 확인 |
| RouterBench 모델이 Claude가 아닌데 라벨을 전이해도 되는가 | capability proxy로 사용할 수 있으나 실제 Claude tier의 충분성을 직접 증명하지 않는다. 소규모 Claude 표본 검증이 필요하다. | 실험 필요 |
| continuous threshold와 RAG 관련 평가를 유지할 것인가 | 3-class baseline 이후 sensitivity 분석 후보로 남아 있다. 서비스 RAG는 구현 범위에서 제외했다. | 실험 필요 |

## 3. RouterArena 평가를 설계하며 해결한 문제

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| RouterArena는 질문별 Bloom 6단계인가 | README는 Bloom 6단계를 설명하지만 공개 Parquet의 질문별 `Difficulty`는 이미 easy/medium/hard 3단계다. | 확인 |
| Bloom 6단계를 2·2·2로 합쳐야 하는가 | 공개 데이터에 질문별 6단계 값이 없으므로 변환하거나 역추정하지 않는다. | 결정 |
| RouterBench와 RouterArena의 같은 label을 정답처럼 비교해도 되는가 | 생성 기준이 다르다. accuracy/F1은 외부 alignment proxy이며 운영 Claude 품질의 직접 증거가 아니다. | 결정 |
| 무엇을 주 분석으로 볼 것인가 | ModernBERT 예측 easy/medium/hard 세 그룹을 만든 뒤 각 그룹 내부 RouterArena label 분포 `P(RouterArena label | predicted group)`를 시각화한다. | 결정 |
| 학습 데이터와 평가 데이터가 겹치면 어떻게 하는가 | 정규화된 질문 hash로 RouterBench 전체 학습 artifact와 exact overlap을 제거한다. 현재 전체 기준 22행이 제외됐다. | 결정 |
| robustness 데이터는 라벨이 없는데 어떻게 평가하는가 | `Global Index`로 full split의 공식 label과 원 질문을 결합해 원본/변형 예측 일관성을 본다. | 결정 |
| 전처리와 평가가 같은 자원을 써야 하는가 | 입력 조립·누수 검사와 RouterArena checkpoint 추론은 Vertex AI Pipelines CPU task로 실행한다. GPU는 ModernBERT CustomJob 학습에만 사용한다. | 결정 |
| GPU 평가가 낭비인가 | GPU가 평가 시간을 줄일 수는 있지만 필수 조건이 “학습만 GPU”이므로 CPU batch·timeout을 조정해 평가하고 Vertex GPU는 학습 종료 후 반납한다. | 결정 |
| 어떤 평가가 아직 남았는가 | 전체 checkpoint 평가, RouterBench holdout evaluator와 robustness consistency가 아직 필요하다. | 실험 필요 |

## 4. ModernBERT 학습과 checkpoint

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| checkpoint 경로는 무엇이며 왜 필요한가 | tokenizer, model weight, config와 label mapping을 저장한 모델 디렉터리다. 평가·재로딩·서빙이 같은 모델 버전을 재현하려면 필요하다. | 확인 |
| `id2label` 순서는 어디서 확인하는가 | checkpoint의 `config.json`과 학습 config/metrics에서 확인한다. 계약은 `0=easy, 1=medium, 2=hard`다. | 결정 |
| Colab 코드와 Kubeflow 학습 코드는 달라야 하는가 | loss·optimizer·metric·label 계약은 공통 학습 코어를 쓴다. 세션이 예고 없이 끊기는 Colab만 별도 wrapper/CLI에서 Google Drive 주기 동기화와 자동 재개를 담당한다. Kubeflow/Vertex 진입점에는 Drive 결합을 넣지 않아 환경별 orchestration만 분리한다. | 결정 |
| 인프라가 없을 때 GPU 학습을 어떻게 하는가 | 100행 CUDA smoke에 더해 전체 RouterBench archive와 전체 학습 notebook을 제공한다. Colab 임시 디스크에서 학습하고, 봉인된 checkpoint를 Drive에 원자적으로 복사하며 새 세션에서 최신 정상 checkpoint를 자동 복구한다. | 구현 완료 |
| 작은 smoke metric을 성능으로 해석해도 되는가 | optimizer step 2 결과의 accuracy 0.3, macro-F1 약 0.288은 학습 성공 판단용이 아니다. CUDA 실행, 저장과 reload 통과가 핵심 결과다. | 확인 |
| 로컬·Colab·Docker runtime을 어떻게 맞추는가 | PyTorch/CUDA/Transformers 등 runtime 계약을 versioned JSON과 metrics에 기록하고 Docker/Vertex 기준 버전을 고정한다. | 결정 |
| T4 전체 학습 설정은 확정인가 | max 8192, FP16, batch 1, accumulation 8, gradient checkpointing은 baseline 시작값이다. 전체 학습 후 early stopping과 checkpoint 선택 정책을 보완한다. | 실험 필요 |
| epoch마다 RouterArena로 좋아지는지 볼 수 있는가 | 3개 epoch 종료 checkpoint마다 같은 RouterArena full을 추론해 predictions·metric·분포 SVG·진행 SVG를 저장한다. RouterArena 결과는 loss·optimizer·early stopping·checkpoint 선택에 들어가지 않고 최종 모델은 고정된 3 epoch 마지막 모델이다. 그래프를 보고 hyperparameter를 바꾸면 RouterArena가 개발 데이터가 되므로 현재는 관찰·보고 전용이다. | 구현 완료 |
| 한 epoch 안에서도 모델이 좋아지는 과정을 볼 수 있는가 | RouterBench validation을 각 epoch의 약 33%·66%·100%에서 측정해 총 9개 점의 loss·accuracy·macro-F1과 SVG를 Drive에 저장한다. RouterArena full은 비용과 외부평가 독립성을 위해 기존처럼 epoch 종료 시 총 3회만 실행한다. 두 관찰 모두 학습·early stopping·checkpoint 선택에 영향을 주지 않는다. | 구현 완료 |
| Colab 무료 세션이 중단되면 어떻게 하는가 | model/tokenizer뿐 아니라 optimizer·scheduler·mixed-precision scaler·RNG·epoch/batch cursor를 보존한다. 로컬 SSD는 50 step, Drive는 500 step과 각 epoch 종료·최종 step에 동기화한다. VM 종료 시 최대 499 step을 재계산하는 대신 대용량 Drive I/O를 줄인다. fingerprint와 checksum을 통과한 checkpoint만 자동 재개한다. | 구현 완료 |
| Colab의 PyTorch 설치가 여러 번 실패한 이유는 무엇인가 | `CalledProcessError`는 겉으로 드러난 결과였고, 실제로는 진단 로그 은폐, Markdown으로 변형된 URL, 선택한 패키지 저장소의 cuDNN 버전 누락이 서로 다른 단계에서 발생했다. 설치 셀을 최소 의존성·명시적 저장소·상세 오류 출력 방식으로 보강했다. | 구현 완료 |
| Vertex/Spot worker가 중단되면 어떻게 하는가 | Colab wrapper는 Google Drive와 `/content`에 특화되어 Vertex에 그대로 적용하지 않는다. 공통 checkpoint 포맷을 이용하되 GCS 동기화·worker 재시작 정책은 별도 구현이 필요하다. | 실험 필요 |

### 4-1. Colab PyTorch 설치 오류를 단계별로 이해하기

이번 오류는 하나의 원인이 반복된 것이 아니라, 앞의 문제를 해결할 때 다음 층의 문제가
드러난 사례다. 컴퓨터 프로그램 오류는 마지막 traceback 한 줄만 보는 것보다 “어느
층까지 정상이고, 다음 층에서 무엇이 실패했는가”를 구분해야 빠르게 해결할 수 있다.

#### 먼저 알아둘 구성요소

고등학교에서 수업을 준비하는 상황에 비유하면 다음과 같다.

| 기술 요소 | 쉬운 비유 | 실제 역할 |
| --- | --- | --- |
| Python notebook | 수업을 진행하는 교실 | 설치 명령과 학습 코드를 실행한다. |
| `subprocess.run` | 다른 담당자에게 심부름을 맡기는 전달자 | notebook에서 별도 `pip` 프로세스를 실행하고 종료 코드를 받는다. |
| pip | 교재 구매 담당자 | 필요한 Python package와 그 dependency를 내려받아 설치한다. |
| package index | 온라인 서점의 상품 목록 | 어떤 package 버전과 wheel 파일을 받을 수 있는지 알려준다. |
| wheel | 운영체제와 Python 버전에 맞게 포장된 완제품 | 소스 빌드 없이 package를 바로 설치하게 한다. |
| PyTorch | 모델 계산 엔진 | tensor 계산, 역전파와 GPU 연산을 담당한다. |
| CUDA | PyTorch가 NVIDIA GPU를 사용하는 통로 | GPU에서 계산을 실행하기 위한 runtime이다. |
| cuDNN | 딥러닝 계산에 특화된 GPU 부품 | attention·행렬 연산 등에 필요한 NVIDIA 신경망 라이브러리다. |

PyTorch 하나를 요청해도 CUDA·cuDNN 등 정확한 버전의 부품이 함께 필요하다. 즉 “교재
한 권”처럼 보여도 실제로는 정해진 판본의 부록 여러 권까지 모두 있어야 설치가 끝난다.

#### 오류 1: `CalledProcessError`만 보여서 실제 원인을 알 수 없었다

처음 보인 메시지는 다음 형태였다.

```text
CalledProcessError: ... pip install ... returned non-zero exit status 1
```

이것은 원인이 아니라 **pip가 실패했다는 결과 보고**다. 심부름을 보낸 담당자가 “구매에
실패했습니다”라고만 알려준 것과 같다. 당시 설치 명령에 `-q`가 있어 pip의 자세한 출력이
줄어들었기 때문에, URL·네트워크·저장 공간·dependency 중 무엇이 문제인지 구분할 수
없었다.

따라서 설치 helper를 다음처럼 바꿨다.

- `-q`를 제거한다.
- pip의 표준 출력과 오류 출력을 함께 수집한다.
- 실패하면 마지막 40줄을 `Last pip output`으로 예외에 포함한다.
- 실행한 전체 명령, Python 경로와 `/content`의 남은 용량도 출력한다.

여기서 얻은 교훈은 **wrapper 예외보다 wrapper가 실행한 프로그램의 원본 로그를 먼저
확보해야 한다**는 것이다.

#### 오류 2: 정상 URL이 Markdown 링크 문자열로 바뀌었다

한 번은 pip에 다음 값이 실제 인자로 전달됐다.

```text
[https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
```

대괄호와 소괄호는 웹 문서에서 클릭 가능한 링크를 만드는 Markdown 문법이다. 사람 눈에는
같은 주소처럼 보이지만, pip 입장에서는 `[`부터 `)`까지가 주소이므로 존재하지 않는
저장소다. 이는 CUDA나 package 버전 문제가 아니라 **코드를 복사·표시하는 과정에서 문자열
자체가 변형된 문제**다.

자동 링크 변환을 피하기 위해 notebook에서는 주소를 다음처럼 두 조각으로 조립한다.

```python
PYTORCH_INDEX = "https:" + "//download.pytorch.org/whl/cu124"
```

여기서 얻은 교훈은 오류 로그의 명령 배열을 보고, 우리가 의도한 값과 실제 프로세스에
전달된 값이 글자 단위로 같은지 확인해야 한다는 것이다.

#### 오류 3: Python의 시스템 보호 가능성을 별도로 대비했다

Colab kernel의 Python 경로가 `/usr/bin/python3`였기 때문에, Debian 계열 Python의
`externally-managed-environment` 보호가 설치를 막을 가능성도 있었다. 이 보호 장치는
운영체제가 관리하는 Python을 pip가 임의로 덮어써 시스템 도구를 망가뜨리지 않게 한다.

이번 최종 로그에서 실제 주원인은 이 보호 장치가 아니었지만, 폐기 가능한 Colab 임시
runtime에서는 아래 옵션을 사용해 이 가능성을 명시적으로 처리했다.

```text
--break-system-packages
```

이 옵션은 개인 PC나 운영 서버에 무조건 사용하면 안 된다. Colab처럼 세션을 버리면 원상
복구되는 환경이라서 허용한 것이다. 즉 **가능성이 있었던 원인과 로그로 확정된 원인을
구분해서 기록해야 한다**.

#### 오류 4: PyTorch는 받았지만 필요한 cuDNN 판본이 저장소에 없었다

상세 로그를 켠 뒤 실제 원인이 다음처럼 확인됐다.

```text
torch-2.5.1+cu124 ... 908.2 MB 다운로드 성공
ERROR: No matching distribution found for
nvidia-cudnn-cu12==9.1.0.70
```

따라서 Python 3.12용 PyTorch wheel이나 인터넷 연결이 문제는 아니었다. PyTorch 본체는
정상적으로 내려받았지만, 선택된 NVIDIA mirror의 상품 목록에는 PyTorch 2.5.1이 정확히
요구하는 cuDNN `9.1.0.70`이 빠져 있었다. 비슷한 `9.1.1.17`이나 최신 `9.x`가 있어도
dependency가 `==9.1.0.70`으로 고정돼 있으므로 pip는 임의로 다른 판본을 고를 수 없다.

공식 PyPI에는 Linux x86-64용 `nvidia-cudnn-cu12==9.1.0.70` wheel이 존재한다. 그래서
PyTorch CUDA wheel은 PyTorch 저장소에서 받고, 누락된 dependency는 PyPI에서도 찾도록
설치 명령을 보강했다.

- PyTorch 2.5.1 CUDA 12.4 공식 설치 조합: [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
- 누락됐던 Linux cuDNN wheel: [PyPI nvidia-cudnn-cu12 9.1.0.70](https://pypi.org/project/nvidia-cudnn-cu12/9.1.0.70/)

```python
PYTORCH_INDEX = "https:" + "//download.pytorch.org/whl/cu124"
PYPI_INDEX = "https:" + "//pypi.org/simple"

install(
    "PyTorch 2.5.1 CUDA 12.4",
    "--index-url", PYTORCH_INDEX,
    "--extra-index-url", PYPI_INDEX,
    "--requirement", "requirements-training-cu124.txt",
)
```

`--index-url`은 기본 서점을 정하고, `--extra-index-url`은 기본 서점에 없는 물건을 찾을
보조 서점을 추가한다. 현재는 모든 핵심 버전을 명시적으로 고정해 의도하지 않은 최신
package가 선택될 가능성을 줄인다.

#### 오류 5: 설치 성공 후 `pyarrow ... size changed`가 발생했다

의존성 설치를 통과한 뒤 Parquet을 읽는 단계에서 다음 오류가 발생했다.

```text
ValueError: pyarrow.lib.IpcReadOptions size changed,
may indicate binary incompatibility.
Expected 112 from C header, got 104 from PyObject
```

이 오류에서 `size`는 Parquet 파일의 크기를 뜻하지 않는다. `pyarrow`의 Python 코드와
내부 C/C++ 확장 모듈이 서로 약속한 메모리 구조의 크기다. 설치 전 버전의 `pyarrow.lib`가
실행 중인 kernel 메모리에 남아 있는데, 디스크의 `_parquet` 파일은 새 버전으로 교체되면
두 부품이 서로 다른 설계도를 보고 대화하게 된다.

이를 교실 비유로 보면 다음과 같다.

- 수업 중인 kernel은 기존 교과서의 페이지 번호를 기억하고 있다.
- pip가 책상 위 교과서를 새 판본으로 교체했다.
- 수업을 계속하면서 옛 페이지 번호로 새 교과서를 찾으니 내용 위치가 맞지 않는다.
- 교실을 다시 시작해 모든 학생이 새 판본을 처음부터 읽게 하는 것이 kernel restart다.

따라서 이 오류의 우선 해결책은 package를 다시 반복 설치하는 것이 아니라 **notebook
kernel을 재시작한 뒤 모든 셀을 처음부터 다시 실행하는 것**이다. `/content`의 설치 파일과
Google Drive 데이터는 kernel 재시작만으로 삭제되지 않지만, Python 변수는 사라지므로
`Run All`로 Drive mount와 경로 변수를 다시 만든다.

노트북에는 재발 방지를 위해 다음 장치를 추가했다.

1. 설치 직후 현재 kernel에서 `torch`나 `pyarrow`를 불러오지 않는다.
2. 별도의 새 Python 프로세스에서 NumPy·Pandas·PyArrow·PyTorch·Transformers import와
   CUDA 사용 가능 여부를 먼저 검증한다.
3. compiled package가 설치·교체되면 현재 process ID를 restart marker에 기록한다.
4. 같은 kernel이 바로 Parquet 셀을 실행하면 학습하기 쉬운 재시작 안내와 함께 차단한다.
5. kernel 재시작 뒤 process ID가 달라지면 marker를 제거하고 Parquet 읽기를 진행한다.

여기서 얻은 교훈은 **디스크에 설치된 버전과 현재 메모리에 이미 적재된 버전은 다를 수
있다**는 것이다. 특히 NumPy·Pandas·PyArrow·PyTorch처럼 C/C++로 작성된 compiled
extension을 실행 중에 교체했다면, 설치 성공만 확인하지 말고 새 프로세스 검증과 kernel
재시작 여부까지 확인해야 한다.

#### 오류 6: 사용하지 않는 `torchvision`이 ModernBERT import를 막았다

kernel 재시작 뒤 학습 CLI까지 실행됐지만 다음 오류가 발생했다.

```text
RuntimeError: operator torchvision::nms does not exist
```

`nms`는 이미지에서 겹치는 여러 탐지 상자를 정리하는 연산이다. ModernBERT는 텍스트
분류기이므로 이 연산을 전혀 사용하지 않는다. 그런데 Colab에는 기존 PyTorch와 함께
설치된 더 새로운 `torchvision`이 남아 있었고, 학습을 위해 PyTorch만 `2.5.1`로 낮추면서
둘의 짝이 맞지 않게 됐다.

Transformers는 여러 종류의 모델을 한 package에서 지원한다. ModernBERT를 불러오는
과정에서도 공통 loss·image utility 모듈을 검사하며, `torchvision`이 설치돼 있으면 선택적
이미지 기능을 import한다. 이때 남아 있던 `torchvision`이 현재 PyTorch에는 없는 방식으로
`torchvision::nms`를 등록하려 해 실패했다. 즉 텍스트 모델이 이미지 연산을 요구한 것이
아니라, **설치돼 있던 선택적 package가 호환되지 않아 import 경로를 오염시킨 것**이다.

해결 방법은 두 가지가 있다.

1. PyTorch 2.5.1과 정확히 맞는 `torchvision==0.20.1`을 설치한다.
2. 이미지 기능을 전혀 사용하지 않으므로 `torchvision`을 제거한다.

현재 프로젝트는 두 번째를 선택했다. 더 적은 package가 다운로드되고 충돌 표면도 줄기
때문이다. 같은 이유로 사용하지 않는 `torchaudio`도 제거한다.

```text
python -m pip uninstall --yes torchvision torchaudio
```

노트북의 fresh-process 검증은 이제 CUDA뿐 아니라 `torchvision`이 실제로 제거됐는지도
확인한다. 여기서 얻은 교훈은 **직접 사용하지 않는 dependency도 설치돼 있다는 이유만으로
자동 import되어 오류를 만들 수 있으므로, ML runtime에는 필요한 package만 남겨야
한다**는 것이다.

#### 함께 적용한 개선과 재발 시 점검 순서

ModernBERT는 이미지·오디오 모델이 아니므로 Colab CUDA requirements에서 사용하지 않는
`torchvision`과 `torchaudio`를 제거했다. 다운로드 시간, 디스크 사용량과 dependency 충돌
가능성을 줄이고 `torch==2.5.1`만 설치한다. `--no-cache-dir`도 사용해 실패한 대용량 wheel이
Colab 임시 디스크에 중복 보관되지 않게 했다.

같은 종류의 오류가 발생하면 다음 순서로 확인한다.

1. traceback의 예외 이름만 보지 말고 `Last pip output`을 확인한다.
2. 출력된 명령에서 URL과 파일 경로가 의도한 문자열인지 확인한다.
3. 다운로드 자체가 실패했는지, 다운로드 후 dependency 해석에서 실패했는지 구분한다.
4. 요구 버전의 wheel이 현재 Python·운영체제·CPU 구조용으로 존재하는지 공식 저장소에서
   확인한다.
5. 기본 index에 없다면 신뢰할 수 있는 보조 index를 추가하되 버전을 고정한다.
6. 학습에 사용하지 않는 package는 설치 목록에서 제거한다.
7. 설치 후 `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`을 출력해
   실제 GPU runtime까지 검증한다.
8. compiled package를 변경했다면 kernel을 재시작하고 `Run All`로 메모리와 디스크의
   package 버전을 일치시킨다.
9. traceback에 사용하지 않는 선택적 package가 등장하면 핵심 의존성과 버전이 맞는지
   확인하고, 필요 없다면 제거한다.

핵심은 “계속 옵션을 하나씩 임의로 추가하는 것”이 아니라, **원본 로그로 실패 층을
확정하고 그 층에만 필요한 수정**을 적용하는 것이다.

## 5. 로컬, Docker, Pod와 Kubeflow 경계

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| 초기 라벨링·전처리도 서버에서 해야 하는가 | 아니다. 초기 데이터 준비와 학습은 Colab에서 공통 core를 호출하고 manifest/checksum을 남긴다. Vertex lineage는 feedback 재학습부터 시작한다. | 2026-09-16 보정 |
| feedback 처리도 로컬에서 하면 되는가 | 반복 자동화, lineage와 재현성이 필요하므로 feedback prepare/merge는 Kubeflow CPU Pod에서 수행한다. | 결정 |
| 같은 CPU image에 로컬 초기 처리와 서버 feedback 처리가 같이 있어도 되는가 | image는 실행 가능한 기능 묶음이고 component가 어떤 CLI를 실행할지 정한다. 같은 image를 재사용해도 Pod와 artifact 경계는 분리된다. | 결정 |
| CPU 작업도 image가 필요한가 | Colab 초기 처리에는 필수가 아니지만 Vertex AI Pipelines의 feedback·qualification component는 재현 가능한 CPU image가 필요하다. | 확인 |
| Pod이 무엇인가 | Kubernetes 실행 단위지만 현행 ML cloud 구조는 GKE Pod를 직접 운영하지 않는다. Vertex AI가 Pipeline task와 CustomJob worker를 관리한다. | 확인 |
| Kubeflow에 라벨링 전용 기능이 있는가 | Kubeflow는 도메인 라벨 알고리즘을 제공하지 않고 실행, artifact, cache, retry와 lineage를 관리한다. 프로젝트 전처리기는 그대로 재사용한다. | 확인 |
| CPU와 GPU 기능을 한 image로 합칠 것인가 | dependency와 실행 위치를 분리한다. data/evaluation/promotion은 Vertex Pipeline CPU image, training은 Vertex AI용 CUDA image로 둔다. 기존 CUDA evaluation image는 CPU용으로 교체한다. | 결정 |
| Docker image와 component는 같은 것인가 | image는 실행 환경, component는 image·명령·입출력·자원 계약, pipeline은 component 실행 순서다. | 확인 |

## 6. GCS, Vertex AI와 cloud 준비

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| GCS 안에 GKE가 있는가 | 아니다. GCS는 artifact 저장소다. 현행 구조는 GKE 자체를 사용하지 않는다. | 확인 |
| CPU/GPU image는 어디에 들어가는가 | 모두 Artifact Registry에 둔다. Vertex Pipeline CPU task는 CPU image를, CustomJob worker는 training CUDA image를 pull하며 데이터·checkpoint·metric은 GCS에 저장한다. | 확인 |
| 라벨링·전처리를 로컬에서 끝내고 결과만 올려도 되는가 | 개발과 빠른 artifact 생성에는 가능하지만 공식 cloud run은 Vertex AI PipelineJob의 CPU component에서 시작한다. 기존 검증 artifact는 KFP importer로 재사용할 수 있다. | 결정 |
| 서버가 없거나 schema가 미정일 때 무엇을 할 수 있는가 | pure core, CLI, fixture, contract validator, Docker smoke, KFP IR compile, 문서와 테스트를 준비할 수 있다. 실제 IAM·GCS 전달은 Vertex cloud smoke가 필요하다. | 확인 |
| “cloud smoke 차단”은 작은 데이터 제한을 푸는 뜻인가 | 아니다. 로컬 구현은 끝났지만 API, bucket, registry, IAM이 없어 실제 Vertex 실행을 못 한다는 뜻이었다. | 확인 |
| 현재 cloud는 준비됐는가 | project와 billing 연결만 확인됐고 GCS bucket, registry와 Vertex Pipeline resource는 아직 미생성이다. architecture와 `us-west1`(2026-08-15 확정, `us-central1`에서 변경)은 확정됐으며 credit 잔액·만료일, resource 이름과 IAM 확인이 남았다. | 확인 |
| 왜 GKE를 사용하지 않는가 | 관리형 Vertex AI Pipelines가 KFP v2 IR과 CPU orchestration을 제공하므로 별도 cluster·control plane·node 운영이 필요하지 않다. 학습만 Vertex CustomJob T4를 사용한다. | 결정 |
| GPU를 무료로 쓸 수 있는가 | 영구 무료 GPU는 아니다. billing 활성화 후 남은 Welcome Credit 안에서 실결제 0원을 목표로 하며 quota·timeout·동시 실행·cleanup guardrail이 필요하다. | 결정 |
| 서버 확보 전 할 수 있는 일을 모두 했는가 | 코드·image smoke·KFP compile 기반은 완료했지만 full evaluator, robustness, resume, gate threshold와 계약 확정은 남아 있다. | 실험 필요 |

## 7. Feedback 재학습과 역할 분담

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| feedback 누적 재학습 코드가 준비됐는가 | 합성 schema 기준 validator, normalize, dedup, RouterBench merge와 CPU→GPU 재학습 pipeline은 구현됐다. 실제 DB export 계약과 cloud 검증은 남았다. | 확인 |
| ML 담당자가 온라인 DB와 API까지 구현해야 하는가 | 아니다. ML은 batch 계약, adapter/validator, 학습·평가·package/gate를 담당하고 온라인 팀이 수집·저장·export·serving을 담당한다. | 결정 |
| feedback schema가 확정되기 전에 image를 만들 수 있는가 | versioned adapter와 합성 fixture로 image를 만들 수 있다. 실제 schema가 바뀌면 adapter와 fixture를 함께 갱신한다. | 결정 |
| 모든 만족 질문을 즉시 학습 라벨로 사용해도 되는가 | 사용자 선호, 일시적 응답 오류와 중복·충돌이 섞이므로 곧바로 채택하면 안 된다. 품질 필터와 반복 근거가 필요하다. | 결정 |
| ML이 escalation 여부를 다시 검사해야 하는가 | upstream export가 유효 feedback만 포함함을 계약으로 보증한다면 비즈니스 규칙을 중복 판단할 필요는 없다. 다만 provenance/audit 필드는 남기고 schema validator는 수행한다. | 팀 협의 |
| 재학습 N은 어떻게 정하는가 | dedup 후 accepted count, class별 최소량, 최대 대기 기간, GPU 비용과 성능 변화를 함께 봐야 한다. 현재 숫자는 미정이다. | 팀 협의 |
| 기존 holdout과 feedback이 겹치면 어떻게 하는가 | validation/test overlap은 제외하고 train과 충돌하면 versioned 정책으로 교체·집계한다. | 결정 |

## 8. 평가, package와 배포 게이트

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| KFP component는 CPU와 GPU 두 개뿐인가 | component는 책임별로 여러 개다. 일반 Vertex Pipeline component는 CPU이고 `modernbert-train`만 T4 CustomJob wrapper로 실행된다. | 확인 |
| RouterArena 예측 그룹은 무엇인가 | ModernBERT가 easy/medium/hard로 예측한 세 집단이다. 각 집단 안의 공식 RouterArena label 분포로 모델의 정렬 특성을 본다. | 확인 |
| model package manifest는 무엇인가 | checkpoint 파일 목록, checksum, label order, model/data/code/evaluation version과 metric 위치를 기록한 인수인계 명세다. | 확인 |
| package validator는 왜 필요한가 | 파일 누락, checksum 손상, 잘못된 label 순서와 평가 증거 누락을 serving 전에 차단한다. | 확인 |
| deploy gate가 바로 서빙에 배포하는가 | 아니다. metric 규칙을 통과한 candidate를 만들고 실패 task를 차단한다. 실제 approved 승격과 serving 전환은 별도 계약이다. | 결정 |
| Kubeflow의 배포 역할은 어디까지인가 | 학습→평가→package→gate와 evidence/lineage 생성까지다. online health check, process 전환과 rollback은 운영 영역이다. | 결정 |
| gate 기준은 확정됐는가 | core는 구현됐지만 실제 baseline과 허용 하락폭은 full 학습·평가 후 정해야 한다. | 실험 필요 |

## 9. 범위에서 제거하거나 다시 정의한 요소

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| RAG가 무엇이며 이 프로젝트에서 필요한가 | 외부 지식을 검색해 생성 입력에 붙이는 구조다. 구현하지 않으며 유사 질문 재사용은 semantic cache가 일부 다른 목적을 수행한다. | 결정 |
| QQP threshold는 ML이 정해야 하는가 | QQP는 캐시된 질문과 새 질문의 의미 동일성 임계값 평가다. 온라인 cache 팀 책임이고 ML DAG에서 제거했다. | 결정 |
| 초기 개요의 Haiku/Sonnet/Opus mapping은 확정인가 | 아니다. Claude 제품 변화에 대응하도록 backend routing config에서 협의·관리해야 한다. | 팀 협의 |
| 초기 개요의 “Kubeflow가 라벨링부터 배포” 표현은 정확한가 | 아니다. 초기 데이터 준비와 ModernBERT 초기학습은 Colab에서 수행한다. 관리형 Vertex AI Pipelines와 GPU CustomJob은 feedback 재학습부터 사용하고, qualification/gate 이후 FastAPI가 CURRENT를 pull한다. | 2026-09-16 보정 |
| MVP 문서와 최종 ML 구조가 다른 것은 충돌인가 | MVP 문서는 역사적 최소 범위다. 현행 구조와 상태는 총괄 체크리스트를 기준으로 하고 세부 명세는 근거 문서로 사용한다. | 결정 |

## 10. 현재 남아 있는 핵심 질문

다음 항목은 코드 기본값과 팀의 운영 결정을 구분해야 한다.

1. 실제 Claude model ID, label mapping, timeout과 fallback
2. feedback export가 보증하는 이벤트 조건과 ML validator의 최소 책임
3. 유효 feedback trigger N, class별 최소량, 최대 대기 기간과 실행 주체
4. full RouterBench/RouterArena baseline과 deploy gate threshold
5. candidate 승인 주체, approved GCS 경로, FastAPI 적용·health check·rollback 방식
6. bucket·registry·cluster·namespace·service account 이름, credit 잔액·만료일과 IAM 담당자
7. CPU RouterArena full 평가의 resource·batch·timeout 실측값
8. Vertex 중단을 견딜 checkpoint resume와 전체 학습 완료 기준

이 질문들의 합의 결과는 `docs/ml/backend-infrastructure-agreement.md`에 기록하고, 실제
진행 상태는 총괄 체크리스트에서 갱신한다.

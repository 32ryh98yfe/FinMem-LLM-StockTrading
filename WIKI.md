# FinMem 프로젝트 Wiki

## 1. 프로젝트 개요

FinMem은 **거대 언어 모델(LLM)을 기반으로 하는 혁신적인 금융 거래 에이전트**입니다. 이 프로젝트의 핵심 목표는 인간 트레이더의 인지 구조를 모방하여, 금융 시장의 복잡하고 변동성 높은 환경 속에서 스스로 학습하고 진화하며 최적의 투자 결정을 내리는 인공지능을 만드는 것입니다.

마치 한 명의 전문 트레이더처럼, FinMem 에이전트는 자신만의 `캐릭터(성격)`를 부여받고, 금융 뉴스, 기업 공시 자료 등 다양한 정보를 `계층화된 기억(Layered Memory)` 속에 저장하고 분석합니다. 이를 통해 과거의 경험으로부터 배우고, 새로운 정보에 민첩하게 반응하며, 지속적으로 투자 전략을 개선해 나갑니다.

## 2. 작동 원리

FinMem의 작동 방식은 다음과 같은 세 가지 핵심 모듈로 구성됩니다.

1.  **프로파일링 (Profiling):** 에이전트에게 특정 `캐릭터`나 투자 성향을 부여합니다. 예를 들어, "공격적인 단기 투자자" 또는 "안정성을 중시하는 장기 투자자"와 같은 역할을 설정할 수 있습니다. 이 캐릭터 설정은 이후 에이전트의 모든 의사 결정 과정에 영향을 미칩니다.

2.  **계층화된 기억 (Memory):** FinMem의 가장 큰 특징으로, 인간의 기억 방식과 유사하게 정보를 세 가지 다른 계층에 저장하고 관리합니다.
    *   **단기 기억 (Short-term Memory):** 최근 발생한 뉴스 기사나 시장 이벤트와 같이 즉각적이고 휘발성이 강한 정보를 저장합니다.
    *   **중기 기억 (Mid-term Memory):** 분기별 기업 실적 보고서(10-Q)와 같이 단기 정보보다는 중요도가 높고, 일정 기간 동안 의사 결정에 영향을 미치는 정보를 저장합니다.
    *   **장기 기억 (Long-term Memory):** 연간 사업 보고서(10-K)나 기업의 근본적인 비즈니스 모델과 같이 오랜 기간 동안 변하지 않는 핵심 정보를 저장합니다.

3.  **의사결정 (Decision-making):** 에이전트는 `계층화된 기억` 속의 정보를 종합적으로 분석하고, 자신의 `캐릭터`에 맞춰 '매수', '매도', '보유'와 같은 투자 결정을 내립니다. 이 과정에서 과거의 투자 결정과 그 결과를 되돌아보는 **'성찰(Reflection)'** 과정을 통해 스스로의 전략을 평가하고 개선합니다.

이러한 과정을 통해 FinMem 에이전트는 마치 살아있는 유기체처럼 금융 환경에 적응하고 진화하게 됩니다.

## 3. 프로젝트 아키텍처

### 파일 구조

프로젝트의 주요 파일 및 디렉토리 구조는 다음과 같습니다.

-   `run.py`: 프로그램의 **시작점**입니다. 시뮬레이션을 실행하고, 환경 설정 및 에이전트 초기화를 담당합니다.
-   `puppy/`: 에이전트의 핵심 로직이 담겨있는 **소스 코드** 디렉토리입니다.
    -   `agent.py`: `LLMAgent` 클래스가 정의된 파일로, 에이전트의 모든 행동을 총괄합니다.
    -   `environment.py`: 주가, 뉴스, 공시 등 시장 데이터를 시뮬레이션하는 환경을 관리합니다.
    -   `memorydb.py`: 단기, 중기, 장기 기억을 관리하는 `BrainDB`가 구현된 파일입니다.
    -   `portfolio.py`: 에이전트의 자산, 거래 내역 등 포트폴리오를 관리합니다.
    -   `reflection.py`: 과거의 경험을 바탕으로 학습하고 전략을 수정하는 '성찰' 기능이 구현되어 있습니다.
-   `config/`: 시뮬레이션에 필요한 각종 설정 파일(예: 에이전트의 캐릭터, 사용할 언어 모델 등)이 위치합니다.
-   `data/`: 시장 데이터, 학습 결과, 로그 등 시뮬레이션 과정에서 생성되는 모든 데이터가 저장됩니다.

## 4. 설치 및 실행 방법

### 환경 설정

1.  **.env 파일 설정:**
    프로젝트를 실행하기 위해서는 OpenAI 또는 HuggingFace의 언어 모델을 사용해야 합니다. 루트 디렉토리에 있는 `.env` 파일을 열어 다음과 같이 자신의 API 키를 입력합니다.

    ```bash
    # OpenAI 모델을 사용하는 경우
    OPENAI_API_KEY = "sk-..."

    # HuggingFace의 TGI 모델을 사용하는 경우
    OPENAI_API_KEY = "sk-..." # 임베딩 모델 용
    HF_TOKEN = "hf_..."
    ```

2.  **config.toml 파일 설정:**
    `config/` 디렉토리의 설정 파일(`config.toml`)에서 사용할 언어 모델의 종류와 세부 설정을 지정할 수 있습니다.

    *   **TGI (HuggingFace) 모델 사용 시:**
        ```toml
        [chat]
        model = "tgi"
        end_point = "<사용할 모델의 엔드포인트 주소>"
        tokenization_model_name = "<모델 이름>"
        ```
    *   **OpenAI 모델 사용 시:**
        ```toml
        [chat]
        model = "gpt-4"
        end_point = "https://api.openai.com/v1/chat/completions"
        tokenization_model_name = "gpt-4"
        ```

### Docker를 이용한 실행

프로젝트는 Docker 환경에서 가장 안정적으로 실행됩니다.

1.  **Docker 이미지 빌드:**
    다음 명령어를 사용하여 Docker 이미지를 빌드합니다.
    ```bash
    docker build -t test-finmem ./.devcontainer
    ```

2.  **Docker 컨테이너 실행:**
    빌드된 이미지를 사용하여 컨테이너를 실행하고, 프로젝트의 루트 폴더로 진입합니다.
    ```bash
    docker run -it --rm -v $(pwd):/finmem test-finmem bash
    ```

### 시뮬레이션 실행

Docker 컨테이너 내부에서 다음 명령어를 사용하여 시뮬레이션을 시작합니다.

-   **학습(Train) 모드 실행:**
    에이전트가 과거 데이터를 학습하며 기억을 형성하는 모드입니다.
    ```bash
    python run.py sim --market-data-path "data/..." --start-time "YYYY-MM-DD" --end-time "YYYY-MM-DD" --run-model "train" --config-path "config/..."
    ```

-   **테스트(Test) 모드 실행:**
    학습된 에이전트를 이용하여 실제 투자 결정을 내리고 성과를 측정하는 모드입니다.
    ```bash
    python run.py sim --market-data-path "data/..." --start-time "YYYY-MM-DD" --end-time "YYYY-MM-DD" --run-model "test" --config-path "config/..." --trained-agent-path "<학습된 에이전트 경로>"
    ```

-   **체크포인트에서 재실행:**
    오류 등으로 시뮬레이션이 중단되었을 경우, 저장된 체크포인트부터 이어서 실행할 수 있습니다.
    ```bash
    python run.py sim-checkpoint --checkpoint-path "<체크포인트 경로>" --run-model "train"
    ```

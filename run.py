# 필요한 라이브러리들을 가져옵니다.
import os  # 운영체제와 상호작용하기 위한 라이브러리 (예: 파일 경로 설정)
import toml  # TOML 형식의 설정 파일을 읽기 위한 라이브러리
import typer  # 사용자 친화적인 명령줄 인터페이스(CLI)를 만들기 위한 라이브러리
import logging  # 프로그램 실행 중 발생하는 이벤트를 기록(로깅)하기 위한 라이브러리
import pickle  # 파이썬 객체를 파일로 저장하거나 불러오기 위한 라이브러리
import warnings  # 경고 메시지를 제어하기 위한 라이브러리
from tqdm import tqdm  # 작업 진행률을 시각적으로 보여주는 라이브러리
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 불러오기 위한 라이브러리
from datetime import datetime  # 날짜와 시간 관련 작업을 위한 라이브러리
from typing import Union  # 타입 힌팅을 위한 라이브러리 (Union: 여러 타입 허용)
from puppy import MarketEnvironment, LLMAgent, RunMode  # 프로젝트의 핵심 클래스들을 가져옵니다.

# --- 초기 설정 ---

# .env 파일에 저장된 환경 변수(예: API 키)를 불러옵니다.
load_dotenv()
# Typer 애플리케이션을 생성합니다. 'puppy'는 이 CLI 애플리케이션의 이름입니다.
app = typer.Typer(name="puppy")
# 실행 중 발생할 수 있는 특정 경고 메시지를 무시하도록 설정합니다.
warnings.filterwarnings("ignore")


# --- 'sim' 명령어: 새로운 시뮬레이션 시작 ---

# 'sim'이라는 이름의 명령어를 정의합니다. 사용자가 터미널에서 'python run.py sim'을 실행하면 이 함수가 호출됩니다.
@app.command("sim", help="새로운 시뮬레이션을 시작합니다.", rich_help_panel="Simulation")
def sim_func(
    # 시뮬레이션에 필요한 다양한 옵션들을 정의합니다. 사용자는 이 옵션들을 통해 시뮬레이션 환경을 제어할 수 있습니다.
    market_data_info_path: str = typer.Option(
        os.path.join("data", "03_model_input", "amzn.pkl"),  # 기본 시장 데이터 파일 경로
        "-mdp",
        "--market-data-path",
        help="시뮬레이션 환경에 사용될 시장 데이터(.pkl) 파일 경로",
    ),
    start_time: str = typer.Option(
        "2022-08-16", "-st", "--start-time", help="시뮬레이션 시작 날짜"
    ),
    end_time: str = typer.Option(
        "2022-10-04", "-et", "--end-time", help="시뮬레이션 종료 날짜"
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="실행 모드: 'train'(학습) 또는 'test'(테스트)"
    ),
    config_path: str = typer.Option(
        os.path.join("config", "amzn_tgi_config.toml"),  # 기본 설정 파일 경로
        "-cp",
        "--config-path",
        help="에이전트 및 시뮬레이션 설정 파일(.toml) 경로",
    ),
    checkpoint_path: str = typer.Option(
        os.path.join("data", "06_train_checkpoint"),  # 기본 체크포인트 저장 경로
        "-ckp",
        "--checkpoint-path",
        help="시뮬레이션 중간 상태를 저장할 체크포인트 경로",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "05_train_model_output"),  # 기본 결과 저장 경로
        "-rp",
        "--result-path",
        help="시뮬레이션 최종 결과를 저장할 경로",
    ),
    trained_agent_path: Union[str, None] = typer.Option(
        None,  # 기본값은 없음
        "-tap",
        "--trained-agent-path",
        help="'test' 모드에서만 사용되며, 미리 학습된 에이전트의 경로",
    ),
) -> None:
    """
    이 함수는 처음부터 새로운 주식 거래 시뮬레이션을 시작하는 역할을 합니다.
    """
    # 설정 파일(.toml)을 불러와 'config' 변수에 저장합니다.
    config = toml.load(config_path)

    # --- 로깅 설정 ---
    # 로그를 기록할 로거(logger)를 설정합니다.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # INFO 레벨 이상의 로그만 기록합니다.
    # 로그 메시지 형식을 지정합니다 (시간 - 이름 - 로그 레벨 - 메시지).
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 로그를 파일로 저장하기 위한 핸들러를 설정합니다.
    file_handler = logging.FileHandler(
        os.path.join(  # 로그 파일 경로와 이름 설정
            "data",
            "04_model_output_log",
            f'{config["general"]["trading_symbol"]}_run.log',
        ),
        mode="a",  # 'a' 모드는 기존 로그 파일에 이어서 기록합니다.
    )
    file_handler.setFormatter(logging_formatter)  # 핸들러에 로그 형식을 적용합니다.
    logger.addHandler(file_handler)  # 로거에 파일 핸들러를 추가합니다.

    # --- 실행 모드 확인 ---
    # 사용자가 입력한 실행 모드('train' 또는 'test')를 확인하고, 프로그램 내부에서 사용할 변수(run_mode_var)로 변환합니다.
    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        # 'train' 또는 'test'가 아닌 다른 값이 입력되면 오류를 발생시킵니다.
        raise ValueError("실행 모드는 반드시 'train' 또는 'test'여야 합니다.")

    # --- 시뮬레이션 환경 생성 ---
    # 저장된 시장 데이터(.pkl) 파일을 바이너리 읽기 모드('rb')로 엽니다.
    with open(market_data_info_path, "rb") as f:
        env_data_pkl = pickle.load(f)  # pickle을 사용해 파일 내용을 불러옵니다.
    # MarketEnvironment 클래스를 사용하여 시뮬레이션 환경을 생성합니다.
    environment = MarketEnvironment(
        symbol=config["general"]["trading_symbol"],  # 거래할 주식 종목
        env_data_pkl=env_data_pkl,  # 시장 데이터
        start_date=datetime.strptime(start_time, "%Y-%m-%d").date(),  # 시뮬레이션 시작일
        end_date=datetime.strptime(end_time, "%Y-%m-%d").date(),  # 시뮬레이션 종료일
    )

    # --- 에이전트 생성 ---
    # 'train' 모드일 경우, 설정 파일(config)을 바탕으로 새로운 LLM 에이전트를 생성합니다.
    if run_mode_var == RunMode.Train:
        the_agent = LLMAgent.from_config(config)
    # 'test' 모드일 경우, 미리 학습된 에이전트를 체크포인트 경로에서 불러옵니다.
    else:
        the_agent = LLMAgent.load_checkpoint(path=os.path.join(trained_agent_path, "agent_1"))

    # --- 시뮬레이션 루프 시작 ---
    # tqdm을 사용하여 전체 시뮬레이션 길이만큼의 진행 바(progress bar)를 생성합니다.
    pbar = tqdm(total=environment.simulation_length)
    # 무한 루프를 시작합니다. 시뮬레이션이 끝나면 내부에서 break로 종료됩니다.
    while True:
        logger.info(f"단계 {the_agent.counter}")  # 현재 단계를 로그에 기록합니다.
        the_agent.counter += 1  # 에이전트의 내부 카운터를 1 증가시킵니다.
        market_info = environment.step()  # 환경을 한 단계 진행시켜 현재 날짜의 시장 정보를 가져옵니다.
        logger.info(f"날짜 {market_info[0]}")  # 현재 날짜를 로그에 기록합니다.
        logger.info(f"기록 {market_info[-2]}")  # 관련 기록을 로그에 기록합니다.

        # market_info의 마지막 값이 True이면 시뮬레이션이 끝났다는 의미이므로 루프를 종료합니다.
        if market_info[-1]:
            break

        # 에이전트가 현재 시장 정보를 바탕으로 한 단계(하루)를 진행합니다. (학습 또는 테스트)
        the_agent.step(market_info=market_info, run_mode=run_mode_var)
        pbar.update(1)  # 진행 바를 한 칸 업데이트합니다.

        # --- 체크포인트 저장 ---
        # 매 단계마다 현재 에이전트와 환경의 상태를 파일로 저장합니다.
        # 이렇게 하면 중간에 오류가 발생해도 처음부터 다시 시작할 필요가 없습니다.
        the_agent.save_checkpoint(path=checkpoint_path, force=True)  # force=True는 기존 파일이 있으면 덮어씁니다.
        environment.save_checkpoint(path=checkpoint_path, force=True)

    # --- 최종 결과 저장 ---
    # 시뮬레이션 루프가 모두 끝나면, 최종 에이전트와 환경의 상태를 지정된 결과 경로에 저장합니다.
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)


# --- 'sim-checkpoint' 명령어: 체크포인트에서 시뮬레이션 재시작 ---

# 'sim-checkpoint'라는 이름의 명령어를 정의합니다.
@app.command(
    "sim-checkpoint",
    help="체크포인트부터 시뮬레이션을 다시 시작합니다.",
    rich_help_panel="Simulation",
)
def sim_checkpoint(
    # 체크포인트 재시작에 필요한 옵션들을 정의합니다.
    checkpoint_path: str = typer.Option(
        os.path.join("data", "06_train_checkpoint"),
        "-ckp",
        "--checkpoint-path",
        help="불러올 체크포인트 경로",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "05_train_model_output"),
        "-rp",
        "--result-path",
        help="시뮬레이션 최종 결과를 저장할 경로",
    ),
    config_path: str = typer.Option(
        os.path.join("config", "aapl_tgi_config.toml"),
        "-cp",
        "--config-path",
        help="설정 파일(.toml) 경로",
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="실행 모드: 'train'(학습) 또는 'test'(테스트)"
    ),
) -> None:
    """
    이 함수는 이전에 저장된 체크포인트부터 시뮬레이션을 이어서 계속하는 역할을 합니다.
    """
    # 설정 파일, 로깅, 실행 모드 확인 등은 'sim_func'와 동일하게 수행합니다.
    config = toml.load(config_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(
            "data",
            "04_model_output_log",
            f'{config["general"]["trading_symbol"]}_run.log',
        ),
        mode="a",
    )
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)

    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        raise ValueError("실행 모드는 반드시 'train' 또는 'test'여야 합니다.")

    # --- 체크포인트에서 환경 및 에이전트 불러오기 ---
    # 'sim_func'와 달리 새로운 객체를 생성하는 대신, 저장된 파일로부터 상태를 복원합니다.
    environment = MarketEnvironment.load_checkpoint(
        path=os.path.join(checkpoint_path, "env")
    )
    the_agent = LLMAgent.load_checkpoint(path=os.path.join(checkpoint_path, "agent_1"))

    # tqdm을 사용하여 진행 바를 생성합니다.
    pbar = tqdm(total=environment.simulation_length)

    # --- 시뮬레이션 루프 시작 ---
    # 이후의 시뮬레이션 루프, 단계별 진행, 체크포인트 저장, 최종 결과 저장 과정은 'sim_func'와 동일합니다.
    while True:
        logger.info(f"단계 {the_agent.counter}")
        the_agent.counter += 1
        market_info = environment.step()
        if market_info[-1]:
            break
        the_agent.step(market_info=market_info, run_mode=run_mode_var)
        pbar.update(1)

        # 매 단계마다 체크포인트를 저장합니다.
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)

    # 시뮬레이션이 완료되면 최종 결과를 저장합니다.
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)


# --- 스크립트 실행 ---

# 이 스크립트가 직접 실행될 때 (예: 'python run.py ...')만 app() 함수를 호출하여
# Typer CLI 애플리케이션을 실행합니다.
if __name__ == "__main__":
    app()

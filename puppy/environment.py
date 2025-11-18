# 필요한 라이브러리들을 가져옵니다.
import os  # 운영체제와 상호작용하기 위한 라이브러리 (파일 경로 등)
import shutil  # 파일 및 폴더 관리를 위한 라이브러리 (삭제 등)
import pickle  # 파이썬 객체를 파일로 저장하거나 불러오기 위한 라이브러리
from datetime import date  # 날짜 정보를 다루기 위한 라이브러리
from typing import List, Dict, Tuple, Union, Any  # 타입 힌팅을 위한 라이브러리
from pydantic import BaseModel, ValidationError  # 데이터 유효성 검사를 위한 라이브러리

# --- 타입 별칭(Type Alias) 정의 ---
# 복잡한 데이터 타입을 간결하게 표현하기 위해 별칭을 만듭니다.

# 에이전트에게 전달될 시장 정보의 데이터 구조를 정의합니다.
market_info_type = Tuple[
    date,               # 현재 날짜
    float,              # 현재 주가
    Union[str, None],   # 현재 날짜의 연간 보고서(10-K), 없으면 None
    Union[str, None],   # 현재 날짜의 분기 보고서(10-Q), 없으면 None
    List[str],          # 현재 날짜의 뉴스 목록
    float,              # '학습' 모드에서 사용할 미래 주가 변동 정보
    bool,               # 시뮬레이션 종료 여부 플래그 (True이면 종료)
]

# 시뮬레이션이 종료되었을 때 반환되는 데이터 구조를 정의합니다.
terminated_market_info_type = Tuple[None, None, None, None, None, None, bool]


# --- 데이터 구조 유효성 검사 ---

class OneDateRecord(BaseModel):
    """
    하루치 시장 데이터가 가져야 할 구조를 Pydantic 모델로 정의하여 데이터의 무결성을 보장합니다.
    이 구조에 맞지 않는 데이터가 들어오면 오류를 발생시킵니다.
    """
    price: Dict[str, float]       # 종목 심볼(str)을 키로, 주가(float)를 값으로 하는 딕셔너리
    filing_k: Dict[str, str]      # 연간 보고서(10-K)
    filing_q: Dict[str, str]      # 분기 보고서(10-Q)
    news: Dict[str, List[str]]    # 뉴스 목록


class MarketEnvironment:
    """
    주식 시장을 시뮬레이션하는 환경 클래스입니다.
    미리 준비된 시장 데이터를 바탕으로 하루씩 시간을 진행시키며,
    각 시점의 주가, 뉴스, 기업 공시 등의 정보를 에이전트에게 제공하는 역할을 합니다.
    """
    def __init__(
        self,
        env_data_pkl: Dict[date, Dict[str, Any]],  # 날짜를 키로 갖는 시장 데이터
        start_date: date,                          # 시뮬레이션 시작일
        end_date: date,                            # 시뮬레이션 종료일
        symbol: str,                               # 거래할 주식 종목 심볼
    ) -> None:
        # --- 데이터 유효성 검사 ---
        # 입력된 데이터의 첫 번째 날짜를 가져옵니다.
        first_date = list(env_data_pkl.keys())[0]
        # 데이터의 키가 'date' 타입이 아니면 오류를 발생시킵니다.
        if not isinstance(first_date, date):
            raise TypeError("환경 데이터(env_data_pkl)의 키는 'date' 타입이어야 합니다.")
        try:
            # 데이터의 구조가 미리 정의한 OneDateRecord 모델과 일치하는지 검사합니다.
            OneDateRecord.model_validate(env_data_pkl[first_date])
        except ValidationError as e:
            # 구조가 일치하지 않으면 오류를 발생시킵니다.
            raise e

        # --- 시뮬레이션 기간 설정 ---
        all_dates = env_data_pkl.keys()
        # 시작일과 종료일이 데이터에 포함되어 있는지 확인합니다.
        if (start_date not in all_dates) or (end_date not in all_dates):
            raise ValueError("시작일과 종료일은 반드시 환경 데이터 내에 포함되어야 합니다.")

        # 시뮬레이션에 사용할 날짜들을 시작일과 종료일 사이로 필터링합니다.
        self.date_series = [
            d for d in all_dates if start_date <= d <= end_date
        ]
        self.date_series = sorted(self.date_series)  # 날짜를 시간순으로 정렬합니다.
        self.date_series_keep = self.date_series.copy()  # reset을 위해 원본을 복사해둡니다.

        self.simulation_length = len(self.date_series) - 1  # 전체 시뮬레이션 길이
        self.start_date = start_date
        self.end_date = end_date
        self.cur_date = None  # 현재 시뮬레이션 날짜 (초기에는 없음)
        self.env_data = env_data_pkl  # 전체 시장 데이터
        self.symbol = symbol  # 거래 종목

    def reset(self) -> None:
        """시뮬레이션 환경을 초기 상태로 되돌립니다."""
        self.date_series = self.date_series_keep.copy()
        self.cur_date = None

    def step(self) -> Union[market_info_type, terminated_market_info_type]:
        """
        시뮬레이션을 하루 진행시킵니다.
        :return: 현재 날짜의 시장 정보 또는 시뮬레이션 종료 신호
        """
        try:
            # date_series 리스트에서 맨 앞의 날짜를 꺼내 현재 날짜로 설정합니다.
            self.cur_date = self.date_series.pop(0)
            # 다음 날짜를 가져옵니다. (미래 주가 변동을 계산하기 위함)
            future_date = self.date_series[0]
        except IndexError:
            # 더 이상 진행할 날짜가 없으면 (리스트가 비어있으면) 시뮬레이션 종료 신호를 반환합니다.
            return (None, None, None, None, None, None, True)

        # --- 현재 날짜의 시장 정보 추출 ---
        cur_date = self.cur_date
        cur_price = self.env_data[self.cur_date]["price"]
        future_price = self.env_data[future_date]["price"]
        cur_filing_k = self.env_data[self.cur_date]["filing_k"]
        cur_filing_q = self.env_data[self.cur_date]["filing_q"]
        cur_news = self.env_data[self.cur_date].get("news", {self.symbol: []})

        # '학습' 모드에서 사용할 정답 데이터 (미래 가격 변동)를 계산합니다.
        cur_record = {
            s: future_price[s] - cur_price[s] for s in cur_price
        }

        # --- 데이터 형식 정리 ---
        # 공시 자료가 없는 경우 None으로 처리합니다.
        final_filing_k = cur_filing_k.get(self.symbol)
        final_filing_q = cur_filing_q.get(self.symbol)

        # 뉴스 자료가 없는 경우 빈 리스트로 처리합니다.
        final_news = cur_news if cur_news else {self.symbol: []}

        # 최종적으로 정리된 시장 정보를 튜플 형태로 반환합니다.
        return (
            cur_date,
            cur_price[self.symbol],
            final_filing_k,
            final_filing_q,
            final_news[self.symbol],
            cur_record[self.symbol],
            False,  # 시뮬레이션이 아직 끝나지 않았음을 알림
        )

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        """현재 환경의 상태(남은 날짜 등)를 파일로 저장(체크포인트)합니다."""
        path = os.path.join(path, "env")
        if os.path.exists(path):
            if force:
                shutil.rmtree(path)  # 기존 파일이 있으면 덮어쓰기
            else:
                raise FileExistsError(f"경로 {path}가 이미 존재합니다.")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "env.pkl"), "wb") as f:
            pickle.dump(self, f)  # 클래스 인스턴스 전체를 저장

    @classmethod
    def load_checkpoint(cls, path: str) -> "MarketEnvironment":
        """저장된 체크포인트 파일로부터 환경 상태를 복원합니다."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"경로 {path}가 존재하지 않습니다.")
        with open(os.path.join(path, "env.pkl"), "rb") as f:
            env = pickle.load(f)
        # 불러온 후, 남은 시뮬레이션 길이를 다시 계산합니다.
        env.simulation_length = len(env.date_series)
        return env

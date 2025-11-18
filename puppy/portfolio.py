# 필요한 라이브러리들을 가져옵니다.
import polars as pl  # 데이터프레임 처리를 위한 고성능 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
from datetime import date  # 날짜 정보를 다루기 위한 라이브러리
from annotated_types import Gt  # Pydantic에서 값의 조건을 지정하기 위한 타입 (Gt: Greater than)
from typing import Dict, Annotated, Union  # 타입 힌팅을 위한 라이브러리
from pydantic import BaseModel  # 데이터 유효성 검사를 위한 라이브러리

class PriceStructure(BaseModel):
    """
    주가 데이터의 유효성을 검사하기 위한 Pydantic 모델입니다.
    주가는 0보다 커야 한다는 규칙을 강제합니다.
    """
    price: Annotated[float, Gt(0)]  # 'price'는 0보다 큰 float 타입이어야 함

class Portfolio:
    """
    에이전트의 주식 포트폴리오를 관리하는 클래스입니다.
    보유 주식 수량, 거래 내역, 시장 가격 변동 등을 기록하고,
    과거 성과를 바탕으로 피드백을 생성하는 역할을 합니다.
    """
    def __init__(self, symbol: str, lookback_window_size: int = 7) -> None:
        # --- 포트폴리오 속성 초기화 ---
        self.cur_date = None  # 현재 날짜
        self.symbol = symbol  # 거래 대상 주식 종목
        self.action_series = {}  # 날짜를 키로, 거래 행동(1: 매수, -1: 매도)을 값으로 저장하는 딕셔너리
        self.market_price = None  # 현재 시장 가격
        self.day_count = 0  # 시뮬레이션이 진행된 총 일수
        self.date_series = []  # 날짜 기록 리스트
        self.holding_shares = 0  # 현재 보유 주식 수량

        # --- 시계열 데이터 저장을 위한 numpy 배열 ---
        self.market_price_series = np.array([])  # 일별 시장 가격 시계열
        self.portfolio_share_series = np.array([])  # 일별 보유 주식 수량 시계열

        # 피드백 계산 시 과거 며칠의 데이터를 볼 것인지 결정
        self.lookback_window_size = lookback_window_size

    def update_market_info(self, new_market_price_info: float, cur_date: date) -> None:
        """
        새로운 날짜의 시장 정보를 업데이트합니다.
        :param new_market_price_info: 새로운 시장 가격
        :param cur_date: 현재 날짜
        """
        # 입력된 가격이 유효한지(0보다 큰지) 검사합니다.
        PriceStructure.model_validate({"price": new_market_price_info})

        # 속성 업데이트
        self.market_price = new_market_price_info
        self.cur_date = cur_date
        self.date_series.append(cur_date)
        self.day_count += 1

        # 시장 가격 시계열 데이터에 새로운 가격을 추가합니다.
        self.market_price_series = np.append(self.market_price_series, new_market_price_info)

    def record_action(self, action: Dict[str, int]) -> None:
        """
        에이전트가 취한 거래 행동(매수/매도)을 기록합니다.
        :param action: {'direction': 1(매수)/-1(매도)} 형태의 딕셔너리
        """
        # 보유 주식 수량을 업데이트합니다.
        self.holding_shares += action["direction"]
        # 현재 날짜에 취한 행동을 기록합니다.
        self.action_series[self.cur_date] = action["direction"]

    def get_action_df(self) -> pl.DataFrame:
        """
        기록된 모든 거래 내역을 Polars 데이터프레임 형태로 반환합니다.
        """
        temp_dict = {"date": [], "symbol": [], "direction": []}
        for date, direction in self.action_series.items():
            temp_dict["date"].append(date)
            temp_dict["symbol"].append(self.symbol)
            temp_dict["direction"].append(direction)
        return pl.DataFrame(temp_dict)

    def update_portfolio_series(self) -> None:
        """
        하루가 끝날 때의 최종 보유 주식 수량을 시계열 데이터에 추가합니다.
        """
        self.portfolio_share_series = np.append(self.portfolio_share_series, self.holding_shares)

    def get_feedback_response(self) -> Union[Dict[str, Union[int, date]], None]:
        """
        과거 일정 기간(lookback_window_size) 동안의 투자 성과를 계산하여 피드백을 반환합니다.
        이 피드백은 에이전트의 기억 시스템에서 어떤 기억이 유용했는지를 평가하는 데 사용됩니다.
        :return: {'feedback': 1(수익)/-1(손실)/0(변동없음), 'date': 피드백 대상 날짜} 또는 None
        """
        # 시뮬레이션 기간이 충분히 길지 않으면 피드백을 생성하지 않습니다.
        if self.day_count <= self.lookback_window_size:
            return None

        # 일별 주가 변동분과 전날의 보유 주식 수를 곱하여 일별 손익을 계산합니다.
        # 예: 어제 10주를 가졌는데 오늘 주가가 5달러 오르면, +50달러의 이익
        daily_pnl = np.diff(self.market_price_series) * self.portfolio_share_series[:-1]

        # 지정된 기간(lookback_window_size) 동안의 누적 손익을 계산합니다.
        cumulative_pnl = np.sum(daily_pnl[-self.lookback_window_size:])

        feedback_date = self.date_series[-self.lookback_window_size]

        # 누적 손익에 따라 피드백 값을 결정합니다.
        if cumulative_pnl > 0:
            return {"feedback": 1, "date": feedback_date}  # 수익
        elif cumulative_pnl < 0:
            return {"feedback": -1, "date": feedback_date} # 손실
        else:
            return {"feedback": 0, "date": feedback_date}  # 변동 없음

    def get_moment(self, moment_window: int = 3) -> Union[Dict[str, int], None]:
        """
        최근 일정 기간(moment_window) 동안의 주가 추세(모멘텀)를 계산합니다.
        이 정보는 '테스트' 모드에서 에이전트의 의사결정에 참고 자료로 사용됩니다.
        :return: {'moment': 1(상승)/-1(하락)/0(변동없음), 'date': 모멘텀 계산 시작 날짜} 또는 None
        """
        # 시뮬레이션 기간이 충분히 길지 않으면 모멘텀을 계산하지 않습니다.
        if self.day_count <= moment_window:
            return None

        # 지정된 기간 동안의 총 주가 변동을 계산합니다.
        price_change = np.sum(np.diff(self.market_price_series)[-moment_window:])

        moment_date = self.date_series[-moment_window]

        # 총 주가 변동에 따라 모멘텀 값을 결정합니다.
        if price_change > 0:
            return {"moment": 1, "date": moment_date}  # 상승 추세
        elif price_change < 0:
            return {"moment": -1, "date": moment_date} # 하락 추세
        else:
            return {"moment": 0, "date": moment_date}  # 변동 없음

# 필요한 라이브러리들을 가져옵니다.
import numpy as np  # 수학 계산, 특히 지수 함수(exp)를 사용하기 위해 import
from typing import Tuple  # 타입 힌팅을 위해 Tuple을 가져옵니다.

class ExponentialDecay:
    """
    시간이 지남에 따라 기억의 '최신성' 점수와 '중요도' 점수를 감소시키는
    '시간 쇠퇴(Time Decay)' 효과를 구현하는 클래스입니다.

    이 클래스는 '지수 쇠퇴(Exponential Decay)' 방식을 사용합니다.
    - 최신성 점수: 시간이 지날수록 빠르게 감소합니다.
    - 중요도 점수: 매일 일정한 비율로 서서히 감소합니다.
    """

    def __init__(
        self,
        recency_factor: float = 10.0,
        importance_factor: float = 0.988,
    ):
        """
        클래스 인스턴스를 초기화합니다.

        :param recency_factor: 최신성 점수가 감소하는 속도를 조절하는 값입니다.
                              값이 클수록 더 천천히 감소합니다.
        :param importance_factor: 중요도 점수가 매일 감소하는 비율입니다.
                                 예를 들어 0.988은 매일 점수가 1.2%씩 감소함을 의미합니다.
        """
        self.recency_factor = recency_factor
        self.importance_factor = importance_factor

    def __call__(
        self, important_score: float, delta: float
    ) -> Tuple[float, float, float]:
        """
        이 클래스의 인스턴스가 함수처럼 호출될 때, 쇠퇴 로직을 실행합니다.
        시뮬레이션의 매 시간 단계(step)마다 각 기억에 대해 이 함수가 호출됩니다.

        :param important_score: 현재 기억의 중요도 점수.
        :param delta: 이 기억이 생성된 후 며칠이 지났는지를 나타내는 카운터.
        :return: (새로운 최신성 점수, 새로운 중요도 점수, 업데이트된 delta)를 담은 튜플.
        """
        # 하루가 지났으므로 delta 카운터를 1 증가시킵니다.
        delta += 1

        # 지수 쇠퇴 공식에 따라 새로운 최신성 점수를 계산합니다.
        # 시간이 지날수록(delta가 커질수록) 이 값은 0에 가까워집니다.
        new_recency_score = np.exp(-(delta / self.recency_factor))

        # 현재 중요도 점수에 팩터를 곱하여 새로운 중요도 점수를 계산합니다.
        new_important_score = important_score * self.importance_factor

        return new_recency_score, new_important_score, delta

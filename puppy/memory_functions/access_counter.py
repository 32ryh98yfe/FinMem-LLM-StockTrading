class LinearImportanceScoreChange:
    """
    기억에 대한 '접근 횟수(access counter)' 피드백을 바탕으로
    기억의 '중요도(Importance)' 점수를 업데이트하는 클래스입니다.

    에이전트가 특정 기억을 참고하여 내린 투자 결정이 좋은 성과를 냈을 때,
    그 기억은 '유용했다'고 판단할 수 있습니다. 이 클래스는 이러한 긍정적/부정적 피드백을
    중요도 점수에 반영하는 역할을 합니다.
    """

    def __call__(self, access_counter: int, importance_score: float) -> float:
        """
        이 클래스의 인스턴스가 함수처럼 호출될 때, 중요도 점수를 업데이트하는 로직을 실행합니다.

        현재 구현은 매우 간단한 선형(Linear) 방식입니다:
        - access_counter가 양수(수익 발생)이면, 중요도 점수가 증가합니다.
        - access_counter가 음수(손실 발생)이면, 중요도 점수가 감소합니다.

        :param access_counter: 피드백 값 (수익 시 +1, 손실 시 -1).
        :param importance_score: 현재 기억의 중요도 점수.
        :return: 업데이트된 새로운 중요도 점수.
        """
        # 현재 중요도 점수에 (접근 카운터 * 5) 만큼을 더하거나 빼서 점수를 업데이트합니다.
        # '5'는 피드백의 강도를 조절하는 가중치 역할을 합니다.
        importance_score += access_counter * 5
        return importance_score

class LinearCompoundScore:
    """
    기억의 여러 가지 점수들을 선형적으로 결합하여 최종 점수를 계산하는 클래스입니다.

    이 클래스는 두 가지 주요 역할을 합니다:
    1. '최신성(Recency)' 점수와 '중요도(Importance)' 점수를 합산하여 부분적인 복합 점수를 생성합니다.
    2. 위에서 계산된 복합 점수와 '유사도(Similarity)' 점수를 합산하여,
       기억을 검색(query)할 때 사용될 최종 순위 점수를 계산합니다.
    """

    def recency_and_importance_score(
        self, recency_score: float, importance_score: float
    ) -> float:
        """
        최신성 점수와 중요도 점수를 결합합니다.

        중요도 점수는 보통 1에서 100 사이의 값을 가지므로, 100으로 나누어
        0과 1 사이의 값인 최신성 점수와 스케일을 맞춰줍니다.

        :param recency_score: 기억의 최신성 점수 (보통 0 ~ 1).
        :param importance_score: 기억의 중요도 점수 (보통 1 ~ 100).
        :return: 두 점수가 결합된 부분 복합 점수.
        """
        # 중요도 점수가 100을 초과하지 않도록 보정합니다.
        importance_score = min(importance_score, 100)
        # 최신성 점수와 스케일이 조정된 중요도 점수를 더합니다.
        return recency_score + (importance_score / 100)

    def merge_score(
        self, similarity_score: float, recency_and_importance: float
    ) -> float:
        """
        유사도 점수와 (최신성+중요도) 복합 점수를 결합하여 최종 검색 순위 점수를 계산합니다.

        :param similarity_score: 쿼리 텍스트와 기억 텍스트 간의 벡터 유사도 점수.
        :param recency_and_importance: `recency_and_importance_score` 메서드로 계산된 점수.
        :return: 최종 순위 점수. 이 점수가 높을수록 쿼리와 더 관련성이 높은 기억으로 판단됩니다.
        """
        # 두 점수를 단순 합산하여 최종 점수를 반환합니다.
        return similarity_score + recency_and_importance

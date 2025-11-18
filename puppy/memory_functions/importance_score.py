# 필요한 라이브러리들을 가져옵니다.
import numpy as np  # 수치 계산, 특히 확률에 따른 랜덤 샘플링을 위해 사용
from abc import ABC, abstractmethod  # 추상 클래스를 만들기 위한 라이브러리 (설계도 역할)

# --- 추상 베이스 클래스 정의 ---

class ImportanceScoreInitialization(ABC):
    """
    모든 '중요도 점수 초기화' 클래스들이 따라야 하는 기본 설계도(추상 클래스)입니다.
    이 클래스를 상속받는 클래스는 반드시 __call__ 메서드를 구현해야 합니다.
    이를 통해 다양한 방식의 중요도 점수 초기화 로직을 일관된 인터페이스로 사용할 수 있습니다.
    """
    @abstractmethod
    def __call__(self) -> float:
        """이 메서드는 호출될 때 초기 중요도 점수(float)를 반환해야 합니다."""
        pass

# --- 팩토리 함수 ---

def get_importance_score_initialization_func(
    type: str, memory_layer: str
) -> ImportanceScoreInitialization:
    """
    설정 값('type'과 'memory_layer')에 따라 적절한 중요도 점수 초기화 클래스의
    인스턴스를 생성하여 반환하는 팩토리 함수입니다.

    :param type: 초기화 방식의 종류 (현재는 'sample'만 지원).
    :param memory_layer: 점수를 초기화할 기억 계층 ('short', 'mid', 'long', 'reflection').
    :return: 설정에 맞는 ImportanceScoreInitialization 인스턴스.
    """
    if type == "sample":
        if memory_layer == "short":
            return I_SampleInitialization_Short()
        elif memory_layer == "mid":
            return I_SampleInitialization_Mid()
        elif memory_layer == "long" or memory_layer == "reflection":
            # 장기 기억과 성찰 기억은 같은 초기화 방식을 사용합니다.
            return I_SampleInitialization_Long()
        else:
            raise ValueError(f"'{memory_layer}'는 유효하지 않은 기억 계층입니다.")
    else:
        raise ValueError(f"'{type}'은(는) 유효하지 않은 중요도 점수 초기화 방식입니다.")

# --- 중요도 점수 초기화 구현 클래스들 ---

class I_SampleInitialization_Short(ImportanceScoreInitialization):
    """
    '단기 기억'을 위한 중요도 점수 초기화 클래스입니다.

    단기 기억에 새로 추가되는 정보는 상대적으로 낮은 중요도를 가질 확률이 높도록 설정되었습니다.
    - 50점(낮음)을 받을 확률: 50%
    - 70점(중간)을 받을 확률: 45%
    - 90점(높음)을 받을 확률: 5%
    """
    def __call__(self) -> float:
        probabilities = [0.5, 0.45, 0.05]
        scores = [50.0, 70.0, 90.0]
        # 주어진 확률에 따라 scores 리스트에서 하나의 값을 랜덤하게 선택하여 반환합니다.
        return np.random.choice(scores, p=probabilities)

class I_SampleInitialization_Mid(ImportanceScoreInitialization):
    """
    '중기 기억'을 위한 중요도 점수 초기화 클래스입니다.

    중기 기억의 정보는 중간 정도의 중요도를 가질 확률이 가장 높습니다.
    - 40점(낮음)을 받을 확률: 5%
    - 60점(중간)을 받을 확률: 80%
    - 80점(높음)을 받을 확률: 15%
    """
    def __call__(self) -> float:
        probabilities = [0.05, 0.8, 0.15]
        scores = [40.0, 60.0, 80.0]
        return np.random.choice(scores, p=probabilities)

class I_SampleInitialization_Long(ImportanceScoreInitialization):
    """
    '장기 기억'과 '성찰 기억'을 위한 중요도 점수 초기화 클래스입니다.

    장기 기억으로 넘어온 정보는 이미 중요하다고 판단된 정보이므로,
    높은 중요도 점수를 받을 확률이 가장 높게 설정되었습니다.
    - 40점(낮음)을 받을 확률: 5%
    - 60점(중간)을 받을 확률: 15%
    - 80점(높음)을 받을 확률: 80%
    """
    def __call__(self) -> float:
        probabilities = [0.05, 0.15, 0.8]
        scores = [40.0, 60.0, 80.0]
        return np.random.choice(scores, p=probabilities)

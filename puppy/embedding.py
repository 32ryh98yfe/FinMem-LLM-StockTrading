# 필요한 라이브러리들을 가져옵니다.
import os
import numpy as np
from typing import List, Union
from langchain_community.embeddings import OpenAIEmbeddings # LangChain 라이브러리에서 OpenAI 임베딩 모델을 가져옵니다.

class OpenAILongerThanContextEmb:
    """
    OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터로 변환(임베딩)하는 클래스입니다.

    이 클래스의 주요 특징은 입력 텍스트가 OpenAI 모델의 최대 처리 길이(Context)보다 길 경우,
    텍스트를 자동으로 여러 개의 작은 조각(chunk)으로 나누어 각각 임베딩한 후,
    그 결과 벡터들의 평균을 내어 최종 임베딩 벡터를 생성하는 것입니다.
    이를 통해 아무리 긴 텍스트라도 효과적으로 벡터화할 수 있습니다.

    참고: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-ada-002",  # 사용할 OpenAI 임베딩 모델 이름
        chunk_size: int = 5000,  # 한 번에 API로 보낼 최대 토큰(단어 조각) 수
        verbose: bool = False,  # 임베딩 진행률 표시 여부
    ) -> None:
        """
        클래스 인스턴스를 초기화합니다.

        :param openai_api_key: OpenAI API 키. 제공되지 않으면 환경 변수에서 찾습니다.
        :param embedding_model: 사용할 임베딩 모델의 이름.
        :param chunk_size: 텍스트를 나눌 때의 기준이 되는 청크 크기.
        :param verbose: 진행률 표시 여부.
        """
        # API 키를 인자 또는 환경 변수에서 가져옵니다.
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        # LangChain의 OpenAIEmbeddings 클래스를 사용하여 임베딩 모델 객체를 생성합니다.
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.openai_api_key,
            chunk_size=chunk_size,
            show_progress_bar=verbose,
        )

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        실제로 텍스트 임베딩을 수행하는 내부 메서드입니다.

        :param text: 임베딩할 텍스트 또는 텍스트 목록.
        :return: 각 텍스트에 대한 임베딩 벡터 목록.
        """
        # 입력이 단일 문자열이면 리스트로 감싸줍니다.
        if isinstance(text, str):
            text = [text]
        # LangChain의 embed_documents 메서드를 호출하여 임베딩을 수행합니다.
        # 이 메서드가 긴 텍스트를 자동으로 처리해줍니다.
        return self.emb_model.embed_documents(texts=text)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        클래스 인스턴스를 함수처럼 호출할 수 있게 해주는 특별 메서드입니다.
        예: emb_func = OpenAILongerThanContextEmb(); result = emb_func("some text")

        :param text: 임베딩할 텍스트 또는 텍스트 목록.
        :return: 임베딩 결과를 담은 NumPy 배열. Faiss와 같은 라이브러리에서 사용하기 용이한 형태입니다.
        """
        # _emb 메서드를 호출하고, 결과를 float32 타입의 NumPy 배열로 변환하여 반환합니다.
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self) -> int:
        """
        현재 사용 중인 임베딩 모델의 벡터 차원 수를 반환합니다.

        벡터의 차원 수는 Faiss 같은 벡터 데이터베이스를 초기화할 때 반드시 필요합니다.

        :return: 임베딩 벡터의 차원 수 (정수).
        :raises NotImplementedError: 지원하지 않는 모델일 경우 발생하는 예외.
        """
        # 모델 이름에 따라 정해진 차원 수를 반환합니다.
        if self.emb_model.model == "text-embedding-ada-002":
            return 1536
        else:
            # 새로운 모델을 추가하려면 이곳에 해당 모델의 차원 수를 추가해야 합니다.
            raise NotImplementedError(
                f"모델 {self.emb_model.model}의 임베딩 차원 수가 정의되지 않았습니다."
            )

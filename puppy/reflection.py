# 필요한 라이브러리 및 모듈들을 가져옵니다.
import logging
import guardrails as gd  # LLM의 출력을 특정 형식으로 강제하고 유효성을 검사하는 라이브.
from datetime import date
from .run_type import RunMode
from pydantic import BaseModel, Field # 데이터 구조를 정의하고 검증하기 위한 라이브러리
from guardrails.validators import ValidChoices # Guardrails에서 선택지를 제한하는 유효성 검사기
from typing import List, Callable, Dict, Union, Any, Tuple
from .chat import LongerThanContextError
from .prompts import ( # LLM에게 보낼 프롬프트 템플릿들을 가져옵니다.
    short_memory_id_desc, mid_memory_id_desc, long_memory_id_desc,
    reflection_memory_id_desc, train_prompt, train_memory_id_extract_prompt,
    train_trade_reason_summary, train_investment_info_prefix,
    test_prompt, test_trade_reason_summary, test_memory_id_extract_prompt,
    test_invest_action_choice, test_investment_info_prefix,
    test_sentiment_explanation, test_momentum_explanation,
)

# --- Pydantic 모델 팩토리 함수 ---
# LLM의 출력 구조를 동적으로 생성하기 위한 함수들입니다.

def _memory_factory(memory_layer: str, id_list: List[int], is_train: bool):
    """
    각 기억 계층(단기, 중기 등)에서 LLM이 어떤 기억을 참고했는지
    ID 목록을 추출하기 위한 Pydantic 모델을 동적으로 생성합니다.
    """
    class Memory(BaseModel):
        # LLM이 'memory_index'라는 필드로 값을 출력하도록 강제합니다.
        memory_index: int = Field(
            ...,
            description=(train_memory_id_extract_prompt if is_train else test_memory_id_extract_prompt).format(
                memory_layer=memory_layer
            ),
            # LLM이 주어진 id_list에 포함된 ID값만 출력하도록 강제합니다.
            # on_fail='reask'는 잘못된 값을 출력하면 다시 물어보라는 의미입니다 (학습 모드에서만).
            validators=[ValidChoices(id_list, on_fail="reask" if is_train else "noop")],
        )
    return Memory

def _reflection_factory(id_lists: Dict[str, List[int]], is_train: bool):
    """
    '성찰'의 전체 결과(참고한 기억 ID 목록, 요약 이유, 투자 결정 등)를
    담을 Pydantic 모델을 동적으로 생성합니다.
    """
    # 각 기억 계층별로 Memory 모델을 생성합니다.
    ShortMem = _memory_factory("short-level", id_lists["short"], is_train)
    MidMem = _memory_factory("mid-level", id_lists["mid"], is_train)
    LongMem = _memory_factory("long-level", id_lists["long"], is_train)
    ReflectionMem = _memory_factory("reflection-level", id_lists["reflection"], is_train)

    class InvestInfo(BaseModel):
        # '테스트' 모드에서는 'investment_decision' 필드를 추가하여
        # 'buy', 'sell', 'hold' 중 하나를 반드시 출력하도록 강제합니다.
        if not is_train:
            investment_decision: str = Field(
                ...,
                description=test_invest_action_choice,
                validators=[ValidChoices(choices=["buy", "sell", "hold"])],
            )

        # 요약 이유 필드는 항상 포함됩니다.
        summary_reason: str = Field(
            ...,
            description=train_trade_reason_summary if is_train else test_trade_reason_summary,
        )

        # 각 기억 계층에 대한 참조 ID 목록 필드를 동적으로 추가합니다.
        # 해당 계층에 기억이 있을 때만 (id_lists에 ID가 있을 때만) 필드를 생성합니다.
        if id_lists["short"]:
            short_memory_index: List[ShortMem] = Field(..., description=short_memory_id_desc)
        if id_lists["mid"]:
            middle_memory_index: List[MidMem] = Field(..., description=mid_memory_id_desc)
        if id_lists["long"]:
            long_memory_index: List[LongMem] = Field(..., description=long_memory_id_desc)
        if id_lists["reflection"]:
            reflection_memory_index: List[ReflectionMem] = Field(..., description=reflection_memory_id_desc)

    return InvestInfo

# --- 헬퍼 함수 ---

def _format_memories(**kwargs) -> Dict[str, Union[List[str], List[int]]]:
    """
    각 기억 계층의 메모리가 비어있거나 하나만 있을 경우,
    Guardrails 유효성 검사기가 정상 작동하도록 임시 데이터를 추가하거나 복제합니다.
    """
    formatted = {}
    for name in ["short", "mid", "long", "reflection"]:
        memory = kwargs.get(f"{name}_memory")
        ids = kwargs.get(f"{name}_memory_id")

        if not memory:
            formatted[f"{name}_memory"] = [f"No {name}-term information.", f"No {name}-term information."]
            formatted[f"{name}_memory_id"] = [-1, -1]
        elif len(memory) == 1:
            formatted[f"{name}_memory"] = [memory[0], memory[0]]
            formatted[f"{name}_memory_id"] = [ids[0], ids[0]]
        else:
            formatted[f"{name}_memory"] = memory
            formatted[f"{name}_memory_id"] = ids

    return formatted

def _delete_placeholder_info(validated_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM의 최종 출력에서 _format_memories 함수가 추가했던
    임시 데이터(-1)를 다시 제거하여 깨끗한 결과만 남깁니다.
    """
    if not validated_output: return {}
    for key in list(validated_output.keys()):
        if "memory_index" in key and validated_output[key] and validated_output[key][0].get("memory_index") == -1:
            del validated_output[key]
    return validated_output

def _build_investment_info_prompt(**kwargs) -> str:
    """
    LLM에게 전달할 전체 프롬프트의 'investment_info' 부분을 구성합니다.
    여기에는 날짜, 종목 정보, 각 기억 계층의 내용 등이 포함됩니다.
    """
    is_train = kwargs["run_mode"] == RunMode.Train

    # 모드에 따라 프롬프트의 접두사를 설정합니다.
    if is_train:
        info = train_investment_info_prefix.format(
            cur_date=kwargs["cur_date"], symbol=kwargs["symbol"], future_record=kwargs["future_record"]
        )
    else:
        info = test_investment_info_prefix.format(
            symbol=kwargs["symbol"], cur_date=kwargs["cur_date"]
        )

    # 각 기억 계층의 내용을 프롬프트에 추가합니다.
    memory_map = {"short": "The short-term information:\n", "mid": "The mid-term information:\n",
                  "long": "The long-term information:\n", "reflection": "The reflection-term information:\n"}

    for name, prefix in memory_map.items():
        if kwargs.get(f"{name}_memory_id") and kwargs[f"{name}_memory_id"][0] != -1:
            info += prefix
            info += "\n".join([f"{i}. {m.strip()}" for i, m in zip(kwargs[f"{name}_memory_id"], kwargs[f"{name}_memory"])])
            if not is_train and name == "short":
                 info += test_sentiment_explanation # 테스트 모드 단기기억에는 감성분석 설명을 추가
            info += "\n\n"

    # 테스트 모드에서는 모멘텀 정보를 추가합니다.
    if not is_train and kwargs.get("momentum") is not None:
        info += test_momentum_explanation
        momentum = kwargs["momentum"]
        if momentum == 1: info += "The cumulative return of past 3 days for this stock is positive."
        elif momentum == -1: info += "The cumulative return of past 3 days for this stock is negative."
        else: info += "The cumulative return of past 3 days for this stock is zero."

    return info

# --- 메인 함수 ---

def trading_reflection(**kwargs) -> Dict[str, Any]:
    """
    에이전트의 '성찰' 과정을 총괄하는 메인 함수입니다.

    1. 입력된 기억 정보들을 포맷팅합니다.
    2. 실행 모드(학습/테스트)에 맞는 Pydantic 모델과 프롬프트를 생성합니다.
    3. Guardrails를 사용하여 LLM에게 요청을 보내고, 출력 형식을 강제합니다.
    4. LLM의 최종 출력을 정리하여 반환합니다.
    """
    run_mode = kwargs["run_mode"]
    logger = kwargs["logger"]

    # 1. 기억 정보 포맷팅
    memories = _format_memories(
        short_memory=kwargs.get("short_memory"), short_memory_id=kwargs.get("short_memory_id"),
        mid_memory=kwargs.get("mid_memory"), mid_memory_id=kwargs.get("mid_memory_id"),
        long_memory=kwargs.get("long_memory"), long_memory_id=kwargs.get("long_memory_id"),
        reflection_memory=kwargs.get("reflection_memory"), reflection_memory_id=kwargs.get("reflection_memory_id"),
    )

    # 2. 모드에 맞는 Pydantic 모델과 프롬프트 생성
    is_train = run_mode == RunMode.Train
    id_lists = {
        "short": memories["short_memory_id"], "mid": memories["mid_memory_id"],
        "long": memories["long_memory_id"], "reflection": memories["reflection_memory_id"]
    }
    response_model = _reflection_factory(id_lists, is_train)
    investment_info = _build_investment_info_prompt(run_mode=run_mode, **kwargs, **memories)

    # 3. Guardrails 설정 및 LLM 호출
    guard = gd.Guard.from_pydantic(
        output_class=response_model,
        prompt=train_prompt if is_train else test_prompt,
        num_reasks=1 # 실패 시 1번까지 다시 질문
    )

    try:
        validated_outcomes = guard(
            kwargs["endpoint_func"],
            prompt_params={"investment_info": investment_info},
        )

        # LLM의 출력이 유효하지 않은 경우, 기본값(hold) 또는 오류 메시지를 반환
        if validated_outcomes.validated_output is None or not isinstance(validated_outcomes.validated_output, dict):
            error_msg = "Guardrails validation failed."
            if validated_outcomes.reask and validated_outcomes.reask.fail_results:
                error_msg = validated_outcomes.reask.fail_results[0].error_message

            logger.info(f"Reflection failed for {kwargs['symbol']}: {error_msg}")

            base_output = {"summary_reason": error_msg}
            if not is_train:
                base_output["investment_decision"] = "hold"
            return base_output

        # 4. 최종 출력 정리 및 반환
        return _delete_placeholder_info(validated_outcomes.validated_output)

    except Exception as e:
        # LLM 호출 중 발생할 수 있는 예외 처리
        if isinstance(e.__context__, LongerThanContextError):
            raise LongerThanContextError from e
        logger.error(f"An exception occurred during reflection: {e}")
        return {}

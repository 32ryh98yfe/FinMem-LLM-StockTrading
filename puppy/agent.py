# 필요한 라이브러리들을 가져옵니다.
import os  # 운영체제와 상호작용하기 위한 라이브러리 (파일/폴더 경로 등)
import shutil  # 파일 및 폴더 관리를 위한 라이브러리 (복사, 삭제 등)
import pickle  # 파이썬 객체를 파일로 저장하거나 불러오기 위한 라이브러리
import logging  # 프로그램 실행 중 발생하는 이벤트를 기록(로깅)하기 위한 라이브러리
from datetime import date  # 날짜 정보를 다루기 위한 라이브러리
from .run_type import RunMode  # 실행 모드(학습/테스트)를 정의한 파일을 가져옵니다.
from .memorydb import BrainDB  # 에이전트의 기억을 관리하는 BrainDB 클래스를 가져옵니다.
from .portfolio import Portfolio  # 포트폴리오 관리를 위한 Portfolio 클래스를 가져옵니다.
from abc import ABC, abstractmethod  # 추상 클래스를 만들기 위한 라이브러리 (설계도 역할)
from .chat import ChatOpenAICompatible  # OpenAI 호환 채팅 모델을 사용하기 위한 클래스를 가져옵니다.
from .environment import market_info_type  # 시장 정보의 데이터 타입을 가져옵니다.
from typing import Dict, Union, Any, List  # 타입 힌팅을 위한 라이브러리
from .reflection import trading_reflection  # '성찰' 기능을 수행하는 함수를 가져옵니다.
from transformers import AutoTokenizer  # HuggingFace의 토크나이저를 사용하기 위한 라이브러리


class TextTruncator:
    """
    LLM에 입력되는 텍스트가 너무 길 경우, 토큰(단어 조각) 수를 제한하여 잘라내는 역할을 하는 클래스입니다.
    LLM은 한 번에 처리할 수 있는 텍스트 길이에 제한이 있기 때문에 이 과정이 필요합니다.
    """
    def __init__(self, tokenization_model_name):
        self.tokenization_model_name = tokenization_model_name
        self.token = os.environ.get("HF_TOKEN", None)  # 환경 변수에서 허깅페이스 토큰을 가져옵니다.
        # 지정된 모델 이름으로 토크나이저를 초기화합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenization_model_name, auth_token=self.token
        )

    def _tokenize_cnt_texts(self, input_text):
        # 텍스트를 토큰화합니다.
        encoded_input = self.tokenizer(input_text)
        # 토큰의 수를 계산합니다.
        num_tokens = len(encoded_input["input_ids"])
        return encoded_input, num_tokens

    def process_list_of_texts(self, list_of_texts, max_total_tokens=320):
        # GPT 모델의 경우 토큰화 방식이 달라 별도 처리가 필요 없을 수 있습니다.
        if "gpt" in self.tokenization_model_name:
            return list_of_texts

        truncated_list = []
        total_tokens = 0
        for text in list_of_texts:
            encoded_input, num_tokens = self._tokenize_cnt_texts(text)

            # 현재까지의 총 토큰 수와 새로운 텍스트의 토큰 수를 더해도 최대치를 넘지 않으면 리스트에 추가합니다.
            if total_tokens + num_tokens <= max_total_tokens:
                truncated_list.append(text)
                total_tokens += num_tokens
            else:
                # 최대치를 넘을 경우, 남은 공간만큼 현재 텍스트를 잘라서 추가하고 종료합니다.
                remaining_tokens = max_total_tokens - total_tokens
                if remaining_tokens > 0:
                    truncated_input_ids = encoded_input["input_ids"][:remaining_tokens]
                    truncated_text = self.tokenizer.decode(
                        truncated_input_ids, skip_special_tokens=True
                    )
                    truncated_list.append(truncated_text)
                    total_tokens += len(truncated_input_ids)
                break
        return truncated_list, total_tokens

    def truncate_text(self, input_text, max_tokens):
        # 단일 텍스트를 최대 토큰 수에 맞춰 잘라냅니다.
        encoded_input, num_tokens = self.tokenize_cnt_texts(input_text)

        if len(encoded_input["input_ids"]) <= max_tokens:
            return input_text, len(encoded_input["input_ids"])

        encoded_input["input_ids"] = encoded_input["input_ids"][:max_tokens]
        encoded_input["attention_mask"] = encoded_input["attention_mask"][:max_tokens]

        output_text = self.tokenizer.decode(encoded_input["input_ids"])
        num_tokens = max_tokens
        return output_text, num_tokens


class Agent(ABC):
    """
    모든 에이전트 클래스의 기본 설계도 역할을 하는 추상 클래스입니다.
    이 클래스를 상속받는 모든 에이전트는 반드시 from_config와 step 메서드를 구현해야 합니다.
    """
    @abstractmethod
    def from_config(self, config: Dict[str, Any]) -> "Agent":
        # 설정 파일로부터 에이전트를 생성하는 메서드
        pass

    @abstractmethod
    def step(self) -> None:
        # 시뮬레이션의 한 단계를 진행하는 메서드
        pass


class LLMAgent(Agent):
    """
    LLM을 기반으로 하는 주식 거래 에이전트의 핵심 클래스입니다.
    """
    def __init__(
        self,
        agent_name: str,
        trading_symbol: str,
        character_string: str,
        brain_db: BrainDB,
        chat_config: Dict[str, Any],
        top_k: int = 1,
        look_back_window_size: int = 7,
    ):
        # --- 기본 속성 초기화 ---
        self.counter = 1  # 시뮬레이션 단계 카운터
        self.top_k = top_k  # 기억을 검색할 때 상위 몇 개를 가져올지 결정
        self.agent_name = agent_name  # 에이전트 이름
        self.trading_symbol = trading_symbol  # 거래할 주식 종목
        self.character_string = character_string  # 에이전트의 캐릭터(투자 성향) 설정
        self.look_back_window_size = look_back_window_size  # 과거 데이터를 얼마나 볼 것인지 결정

        # --- 로깅 설정 ---
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(
            os.path.join("data", "04_model_output_log", f"{self.trading_symbol}_run.log"),
            mode="a",
        )
        file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(file_handler)

        # --- 핵심 모듈 초기화 ---
        self.brain = brain_db  # 기억을 관리하는 BrainDB 객체
        self.portfolio = Portfolio(  # 포트폴리오 관리 객체
            symbol=self.trading_symbol, lookback_window_size=self.look_back_window_size
        )

        # --- LLM 채팅 설정 ---
        self.chat_config_save = chat_config.copy()  # 나중에 저장하기 위해 설정 복사본 저장
        chat_config = chat_config.copy()
        end_point = chat_config["end_point"]
        model = chat_config["model"]
        system_message = chat_config["system_message"]

        # --- 텍스트 길이 제어 설정 ---
        self.model_name = chat_config["model"]
        self.max_token_short = chat_config.get("max_token_short", None)
        self.max_token_mid = chat_config.get("max_token_mid", None)
        self.max_token_long = chat_config.get("max_token_long", None)
        self.max_token_reflection = chat_config.get("max_token_reflection", None)

        del chat_config["end_point"]
        del chat_config["model"]
        del chat_config["system_message"]

        if self.max_token_short:
            # 텍스트 트렁케이터 초기화
            self.truncator = TextTruncator(
                tokenization_model_name=chat_config["tokenization_model_name"]
            )

        # LLM 모델과의 통신을 담당하는 엔드포인트 설정
        self.guardrail_endpoint = ChatOpenAICompatible(
            end_point=end_point,
            model=model,
            system_message=system_message,
            other_parameters=chat_config,
        ).guardrail_endpoint()

        # --- 결과 및 기록 저장을 위한 변수 ---
        self.reflection_result_series_dict = {}  # 날짜별 성찰 결과를 저장
        self.access_counter = {}  # 각 기억에 얼마나 자주 접근했는지 기록

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMAgent":
        """설정 파일(.toml)의 내용으로부터 LLMAgent 객체를 생성합니다."""
        return cls(
            agent_name=config["general"]["agent_name"],
            trading_symbol=config["general"]["trading_symbol"],
            character_string=config["general"]["character_string"],
            brain_db=BrainDB.from_config(config=config),  # BrainDB도 설정 파일로부터 생성
            top_k=config["general"].get("top_k", 5),
            chat_config=config["chat"],
            look_back_window_size=config["general"]["look_back_window_size"],
        )

    def _handling_filings(self, cur_date: date, filing_q: str, filing_k: str) -> None:
        """기업 공시 자료(10-Q, 10-K)를 처리하여 각각 중기, 장기 기억에 저장합니다."""
        if filing_q:  # 분기 보고서(10-Q)가 있으면
            self.brain.add_memory_mid(symbol=self.trading_symbol, date=cur_date, text=filing_q)
        if filing_k:  # 연간 보고서(10-K)가 있으면
            self.brain.add_memory_long(symbol=self.trading_symbol, date=cur_date, text=filing_k)

    def _handling_news(self, cur_date: date, news: List[str]) -> None:
        """뉴스 기사를 처리하여 단기 기억에 저장합니다."""
        if news != []:
            self.brain.add_memory_short(symbol=self.trading_symbol, date=cur_date, text=news)
    
    def __query_info_for_reflection(self, run_mode: RunMode):
        """'성찰'을 위해 필요한 모든 정보(단기/중기/장기/성찰 기억)를 BrainDB에서 가져옵니다."""
        self.logger.info(f"종목: {self.trading_symbol}\n")

        # 단기 기억 조회
        cur_short_queried, cur_short_memory_id = self.brain.query_short(
            query_text=self.character_string, top_k=self.top_k, symbol=self.trading_symbol
        )
        # TGI 모델 사용 시 텍스트 길이 조절
        if self.model_name.startswith("tgi"):
            cur_short_queried, _ = self.truncator.process_list_of_texts(
                cur_short_queried, max_total_tokens=self.max_token_short
            )
        # 조회된 내용 로그 기록
        for cur_id, cur_memory in zip(cur_short_memory_id, cur_short_queried):
            self.logger.info(f"상위 단기기억 {cur_id}: {cur_memory}\n")

        # 중기 기억 조회 (단기 기억과 동일한 패턴)
        cur_mid_queried, cur_mid_memory_id = self.brain.query_mid(
            query_text=self.character_string, top_k=self.top_k, symbol=self.trading_symbol
        )
        if self.model_name.startswith("tgi"):
            cur_mid_queried, _ = self.truncator.process_list_of_texts(
                cur_mid_queried, max_total_tokens=self.max_token_mid
            )
        for cur_id, cur_memory in zip(cur_mid_memory_id, cur_mid_queried):
            self.logger.info(f"상위 중기기억 {cur_id}: {cur_memory}\n")

        # 장기 기억 조회 (단기 기억과 동일한 패턴)
        cur_long_queried, cur_long_memory_id = self.brain.query_long(
            query_text=self.character_string, top_k=self.top_k, symbol=self.trading_symbol
        )
        if self.model_name.startswith("tgi"):
            cur_long_queried, _ = self.truncator.process_list_of_texts(
                cur_long_queried, max_total_tokens=self.max_token_long
            )
        for cur_id, cur_memory in zip(cur_long_memory_id, cur_long_queried):
            self.logger.info(f"상위 장기기억 {cur_id}: {cur_memory}\n")

        # 성찰 기억 조회 (단기 기억과 동일한 패턴)
        cur_reflection_queried, cur_reflection_memory_id = self.brain.query_reflection(
            query_text=self.character_string, top_k=self.top_k, symbol=self.trading_symbol
        )
        if self.model_name.startswith("tgi"):
            cur_reflection_queried, _ = self.truncator.process_list_of_texts(
                cur_reflection_queried, max_total_tokens=self.max_token_reflection
            )
        for cur_id, cur_memory in zip(cur_reflection_memory_id, cur_reflection_queried):
            self.logger.info(f"상위 성찰기억 {cur_id}: {cur_memory}\n")

        # 테스트 모드에서는 추가적으로 주가 모멘텀 정보를 가져옵니다.
        cur_moment = None
        if run_mode == RunMode.Test:
            cur_moment_ret = self.portfolio.get_moment(moment_window=3)
            if cur_moment_ret is not None:
                cur_moment = cur_moment_ret["moment"]

        # 최종적으로 조회된 모든 정보를 반환합니다.
        if run_mode == RunMode.Train:
            return (cur_short_queried, cur_short_memory_id, cur_mid_queried, cur_mid_memory_id,
                    cur_long_queried, cur_long_memory_id, cur_reflection_queried, cur_reflection_memory_id)
        else: # Test 모드
            return (cur_short_queried, cur_short_memory_id, cur_mid_queried, cur_mid_memory_id,
                    cur_long_queried, cur_long_memory_id, cur_reflection_queried, cur_reflection_memory_id, cur_moment)


    def __reflection_on_record(
        self, cur_date: date, run_mode: RunMode, cur_record: Union[float, None] = None
    ) -> Dict[str, Any]:
        """가져온 정보를 바탕으로 LLM에게 추론을 요청하여 '성찰'을 수행합니다."""
        if run_mode == RunMode.Train and not cur_record:
            self.logger.info("기록이 없어 성찰을 수행하지 않습니다.\n")
            return {}

        # 성찰에 필요한 정보를 조회합니다.
        query_results = self.__query_info_for_reflection(run_mode=run_mode)

        # trading_reflection 함수를 호출하여 LLM에게 분석 및 의사결정을 요청합니다.
        reflection_result = trading_reflection(
            cur_date=cur_date,
            symbol=self.trading_symbol,
            run_mode=run_mode,
            endpoint_func=self.guardrail_endpoint,
            short_memory=query_results[0],
            short_memory_id=query_results[1],
            mid_memory=query_results[2],
            mid_memory_id=query_results[3],
            long_memory=query_results[4],
            long_memory_id=query_results[5],
            reflection_memory=query_results[6],
            reflection_memory_id=query_results[7],
            future_record=cur_record if run_mode == RunMode.Train else None,
            momentum=query_results[8] if run_mode == RunMode.Test else None,
            logger=self.logger,
        )

        # LLM으로부터 받은 성찰 결과(요약 및 추론 이유)가 유효하면, 이를 다시 '성찰 기억'으로 저장합니다.
        if reflection_result and "summary_reason" in reflection_result:
            self.brain.add_memory_reflection(
                symbol=self.trading_symbol,
                date=cur_date,
                text=reflection_result["summary_reason"],
            )
        else:
            self.logger.info("성찰 결과가 수렴되지 않아 저장하지 않습니다.\n")

        return reflection_result

    def _reflect(
        self, cur_date: date, run_mode: RunMode, cur_record: Union[float, None] = None
    ) -> None:
        """성찰 과정을 실행하고 결과를 기록합니다."""
        reflection_result_cur_date = self.__reflection_on_record(
            cur_date=cur_date, run_mode=run_mode, cur_record=cur_record
        )
        # 날짜별 성찰 결과를 딕셔너리에 저장합니다.
        self.reflection_result_series_dict[cur_date] = reflection_result_cur_date

        # 모드에 따라 다른 로그를 출력합니다.
        if run_mode == RunMode.Train:
            self.logger.info(
                f"{self.trading_symbol}-날짜 {cur_date}\n성찰 요약: {reflection_result_cur_date.get('summary_reason')}\n\n"
            )
        elif run_mode == RunMode.Test:
            if reflection_result_cur_date:
                self.logger.info(
                    f"!!거래 결정: {reflection_result_cur_date.get('investment_decision')} !! {self.trading_symbol}-날짜 {cur_date}\n투자 이유: {reflection_result_cur_date.get('summary_reason')}\n\n"
                )
            else:
                self.logger.info("결정 없음")

    def _construct_train_actions(self, cur_record: float) -> Dict[str, int]:
        """'학습' 모드에서 취할 행동을 결정합니다. 실제 미래 주가(cur_record)를 보고 정답을 맞추는 방식입니다."""
        cur_direction = 1 if cur_record > 0 else -1  # 주가가 올랐으면 '매수', 내렸으면 '매도'
        return {"direction": cur_direction, "quantity": 1}

    def _portfolio_step(self, cur_action: Dict[str, int]) -> None:
        """결정된 행동을 포트폴리오에 기록하고, 포트폴리오 상태를 업데이트합니다."""
        self.portfolio.record_action(action=cur_action)
        self.portfolio.update_portfolio_series()

    def _update_access_counter(self):
        """포트폴리오의 피드백(수익/손실)을 바탕으로, 의사결정에 사용된 기억들의 중요도를 업데이트합니다."""
        feedback = self.portfolio.get_feedback_response()
        if not feedback or feedback["feedback"] == 0:
            return  # 피드백이 없으면 아무것도 하지 않습니다.

        cur_date = feedback["date"]
        cur_memory = self.reflection_result_series_dict[cur_date]

        # 수익/손실에 기여한 각 기억층(단기/중기/장기/성찰)의 기억들에 접근 횟수를 업데이트합니다.
        # (세부 로직은 __update_access_counter_sub 헬퍼 함수에 위임)
        if "short_memory_index" in cur_memory:
            self.__update_access_counter_sub(cur_memory, "short_memory_index", feedback)
        if "middle_memory_index" in cur_memory:
            self.__update_access_counter_sub(cur_memory, "middle_memory_index", feedback)
        if "long_memory_index" in cur_memory:
            self.__update_access_counter_sub(cur_memory, "long_memory_index", feedback)
        if "reflection_memory_index" in cur_memory:
            self.__update_access_counter_sub(cur_memory, "reflection_memory_index", feedback)

    def __update_access_counter_sub(self, cur_memory, layer_index_name, feedback):
        """기억 접근 횟수를 업데이트하는 헬퍼 함수입니다."""
        if cur_memory[layer_index_name] is not None:
            cur_ids = [i["memory_index"] for i in cur_memory[layer_index_name] if "memory_index" in i]
            unique_ids = list(set(cur_ids))
            self.brain.update_access_count_with_feed_back(
                symbol=self.trading_symbol,
                ids=unique_ids,
                feedback=feedback["feedback"],
            )

    @staticmethod
    def __process_test_action(test_reflection_result: Dict[str, Any]) -> Dict[str, int]:
        """'테스트' 모드에서 LLM의 성찰 결과를 실제 거래 행동('매수'/'매도'/'보유')으로 변환합니다."""
        if not test_reflection_result:
            return {"direction": 0}  # 결과가 없으면 '보유'

        decision = test_reflection_result.get("investment_decision")
        if decision == "buy":
            return {"direction": 1}  # 매수
        elif decision == "hold":
            return {"direction": 0}  # 보유
        else: # sell
            return {"direction": -1} # 매도

    def step(self, market_info: market_info_type, run_mode: RunMode) -> None:
        """
        시뮬레이션의 한 단계(하루)를 진행하는 메인 메서드입니다.
        데이터 처리 -> 성찰 -> 행동 결정 -> 포트폴리오 업데이트의 흐름을 따릅니다.
        """
        # 0. 실행 모드 검증 및 현재 시장 정보 파싱
        if run_mode not in [RunMode.Train, RunMode.Test]:
            raise ValueError("실행 모드는 'Train' 또는 'Test'여야 합니다.")

        cur_date, cur_price, cur_filing_k, cur_filing_q, cur_news, cur_record, _ = market_info

        # 1. 새로운 정보(공시, 뉴스)를 기억에 추가
        self._handling_filings(cur_date=cur_date, filing_q=cur_filing_q, filing_k=cur_filing_k)
        self._handling_news(cur_date=cur_date, news=cur_news)

        # 2. 현재 주가를 포트폴리오에 업데이트
        self.portfolio.update_market_info(new_market_price_info=cur_price, cur_date=cur_date)

        # 3. 모든 정보를 종합하여 '성찰' 수행
        self._reflect(cur_date=cur_date, run_mode=run_mode, cur_record=cur_record)

        # 4. 성찰 결과를 바탕으로 행동 결정
        if run_mode == RunMode.Train:
            cur_action = self._construct_train_actions(cur_record=cur_record)
        else:  # Test 모드
            cur_action = self.__process_test_action(
                test_reflection_result=self.reflection_result_series_dict[cur_date]
            )

        # 5. 결정된 행동을 포트폴리오에 반영
        self._portfolio_step(cur_action=cur_action)

        # 6. 거래 결과에 따라 관련 기억의 중요도 업데이트
        self._update_access_counter()

        # 7. BrainDB의 내부 상태 업데이트 (예: 오래된 기억 처리)
        self.brain.step()

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        """현재 에이전트의 모든 상태(기억, 포트폴리오 등)를 파일로 저장(체크포인트)합니다."""
        path = os.path.join(path, self.agent_name)
        if os.path.exists(path):
            if force:
                shutil.rmtree(path)  # 기존 파일이 있으면 덮어쓰기
            else:
                raise FileExistsError(f"경로 {path}가 이미 존재합니다.")
        os.makedirs(os.path.join(path, "brain"), exist_ok=True)

        state_dict = {
            "agent_name": self.agent_name,
            "character_string": self.character_string,
            "top_k": self.top_k,
            "counter": self.counter,
            "trading_symbol": self.trading_symbol,
            "portfolio": self.portfolio,
            "chat_config": self.chat_config_save,
            "reflection_result_series_dict": self.reflection_result_series_dict,
            "access_counter": self.access_counter,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        self.brain.save_checkpoint(path=os.path.join(path, "brain"), force=force)

    @classmethod
    def load_checkpoint(cls, path: str) -> "LLMAgent":
        """저장된 체크포인트 파일로부터 에이전트의 상태를 복원합니다."""
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)

        brain = BrainDB.load_checkpoint(path=os.path.join(path, "brain"))

        class_obj = cls(
            agent_name=state_dict["agent_name"],
            trading_symbol=state_dict["trading_symbol"],
            character_string=state_dict["character_string"],
            brain_db=brain,
            top_k=state_dict["top_k"],
            chat_config=state_dict["chat_config"],
        )
        class_obj.portfolio = state_dict["portfolio"]
        class_obj.reflection_result_series_dict = state_dict["reflection_result_series_dict"]
        class_obj.access_counter = state_dict["access_counter"]
        class_obj.counter = state_dict["counter"]
        return class_obj

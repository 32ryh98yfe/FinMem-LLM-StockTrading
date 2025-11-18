# 필요한 라이브러리들을 가져옵니다.
import os
import faiss  # 벡터 검색을 위한 라이브러리 (유사한 텍스트를 빠르게 찾기 위함)
import pickle
import logging
import shutil
import numpy as np
from datetime import date
from itertools import repeat
from sortedcontainers import SortedList  # 정렬된 상태를 유지하는 리스트
from .embedding import OpenAILongerThanContextEmb  # 텍스트를 벡터로 변환(임베딩)하는 클래스
from typing import List, Union, Dict, Any, Tuple, Callable
from .memory_functions import (  # 기억의 점수(중요도, 최신성 등)를 계산하는 함수들
    ImportanceScoreInitialization,
    get_importance_score_initialization_func,
    R_ConstantInitialization,
    LinearCompoundScore,
    ExponentialDecay,
    LinearImportanceScoreChange,
)

class id_generator_func:
    """각 기억에 고유한 ID를 부여하기 위한 간단한 ID 생성기 클래스입니다."""
    def __init__(self):
        self.current_id = 0

    def __call__(self):
        new_id = self.current_id
        self.current_id += 1
        return new_id

class MemoryDB:
    """
    하나의 기억 계층(예: 단기 기억)을 관리하는 데이터베이스 클래스입니다.
    텍스트 정보를 벡터로 변환하여 저장하고, 중요도/최신성 점수를 관리하며,
    유사도 기반 검색을 지원합니다.
    """
    def __init__(
        self,
        db_name: str,
        id_generator: Callable,
        jump_threshold_upper: float,  # 중요도 점수가 이 값 이상이면 상위 기억층으로 이동
        jump_threshold_lower: float,  # 중요도 점수가 이 값 미만이면 하위 기억층으로 이동
        logger: logging.Logger,
        emb_config: Dict[str, Any],
        # 다양한 점수 계산 방식을 외부에서 주입받아 유연성을 높입니다.
        importance_score_initialization: ImportanceScoreInitialization,
        recency_score_initialization: R_ConstantInitialization,
        compound_score_calculation: LinearCompoundScore,
        importance_score_change_access_counter: LinearImportanceScoreChange,
        decay_function: ExponentialDecay,
        clean_up_threshold_dict: Dict[str, float],  # 기억을 삭제할 임계값
    ) -> None:
        # --- DB 속성 초기화 ---
        self.db_name = db_name
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.emb_config = emb_config
        self.emb_func = OpenAILongerThanContextEmb(**self.emb_config)
        self.emb_dim = self.emb_func.get_embedding_dimension()
        self.logger = logger

        # --- 점수 계산 함수 초기화 ---
        self.importance_score_initialization_func = importance_score_initialization
        self.recency_score_initialization_func = recency_score_initialization
        self.compound_score_calculation_func = compound_score_calculation
        self.decay_function = decay_function
        self.importance_score_change_access_counter = importance_score_change_access_counter
        self.clean_up_threshold_dict = clean_up_threshold_dict

        # 여러 주식 종목의 기억을 관리하기 위한 딕셔너리
        self.universe = {}

    def add_new_symbol(self, symbol: str) -> None:
        """새로운 주식 종목을 위한 저장 공간을 초기화합니다."""
        # Faiss 인덱스 생성 (코사인 유사도 기반 검색)
        cur_index = faiss.IndexFlatIP(self.emb_dim)
        cur_index = faiss.IndexIDMap2(cur_index) # 벡터에 ID를 매핑

        self.universe[symbol] = {
            # 점수를 기준으로 정렬된 기억 목록
            "score_memory": SortedList(key=lambda x: x["important_score_recency_compound_score"]),
            "index": cur_index,  # Faiss 벡터 인덱스
        }

    def add_memory(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        """새로운 텍스트 정보를 기억으로 추가합니다."""
        if symbol not in self.universe:
            self.add_new_symbol(symbol)

        if isinstance(text, str):
            text = [text]

        # 1. 텍스트를 임베딩 벡터로 변환
        emb = self.emb_func(text)
        faiss.normalize_L2(emb)  # 벡터 정규화 (코사인 유사도 계산을 위함)

        # 2. 각 텍스트에 고유 ID 부여
        ids = [self.id_generator() for _ in range(len(text))]

        # 3. 초기 중요도 및 최신성 점수 계산
        importance_scores = [self.importance_score_initialization_func() for _ in range(len(text))]
        recency_scores = [self.recency_score_initialization_func() for _ in range(len(text))]

        # 4. 중요도와 최신성을 결합한 부분 점수 계산
        partial_scores = [
            self.compound_score_calculation_func.recency_and_importance_score(r, i)
            for r, i in zip(recency_scores, importance_scores)
        ]

        # 5. Faiss 인덱스에 벡터와 ID 추가
        self.universe[symbol]["index"].add_with_ids(emb, np.array(ids))

        # 6. score_memory 리스트에 기억의 메타데이터 추가
        for i in range(len(text)):
            memory_record = {
                "text": text[i],
                "id": ids[i],
                "important_score": importance_scores[i],
                "recency_score": recency_scores[i],
                "delta": 0,
                "important_score_recency_compound_score": partial_scores[i],
                "access_counter": 0,  # 접근 횟수
                "date": date,
            }
            self.universe[symbol]["score_memory"].add(memory_record)
            self.logger.info(memory_record)

    def query(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """
        주어진 텍스트(query_text)와 가장 관련 있는 기억 'top_k'개를 검색합니다.
        검색은 (1)벡터 유사도와 (2)중요도/최신성 점수를 모두 고려하여 수행됩니다.
        """
        if not self.universe.get(symbol) or not self.universe[symbol]["score_memory"] or top_k == 0:
            return [], []

        max_len = len(self.universe[symbol]["score_memory"])
        top_k = min(top_k, max_len)

        cur_index = self.universe[symbol]["index"]
        query_emb = self.emb_func(query_text) # 쿼리 텍스트도 벡터로 변환
        faiss.normalize_L2(query_emb)

        # --- 검색 파트 1: 벡터 유사도 기준 상위 K개 검색 ---
        dists, ids = cur_index.search(query_emb, top_k)

        # --- 검색 파트 2: 중요도/최신성 점수 기준 상위 K개 검색 ---
        top_score_records = self.universe[symbol]["score_memory"][-top_k:]

        # --- 종합: 두 그룹의 후보들을 종합하여 최종 점수를 계산하고 순위를 매김 ---
        candidates = []
        processed_ids = set()

        # 벡터 유사도 기반 후보 추가
        for dist, id_ in zip(dists[0], ids[0]):
            if id_ != -1 and id_ not in processed_ids:
                record = next((r for r in self.universe[symbol]["score_memory"] if r["id"] == id_), None)
                if record:
                    final_score = self.compound_score_calculation_func.merge_score(dist, record["important_score_recency_compound_score"])
                    candidates.append({"record": record, "score": final_score})
                    processed_ids.add(id_)

        # 점수 기반 후보 추가
        for record in top_score_records:
            if record["id"] not in processed_ids:
                vec = cur_index.reconstruct(record["id"]).reshape(1, -1)
                faiss.normalize_L2(vec)
                dist = np.dot(query_emb, vec.T)[0][0]
                final_score = self.compound_score_calculation_func.merge_score(dist, record["important_score_recency_compound_score"])
                candidates.append({"record": record, "score": final_score})
                processed_ids.add(record["id"])

        # 최종 점수 기준 내림차순 정렬
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 상위 K개의 텍스트와 ID를 반환
        top_k_candidates = candidates[:top_k]
        return [c["record"]["text"] for c in top_k_candidates], [c["record"]["id"] for c in top_k_candidates]


    def update_access_count_with_feed_back(self, symbol: str, ids: List[int], feedback: int) -> List[int]:
        """거래 피드백(수익/손실)을 바탕으로, 관련 기억의 접근 횟수와 중요도 점수를 업데이트합니다."""
        if symbol not in self.universe:
            return []

        updated_ids = []
        for record in self.universe[symbol]["score_memory"]:
            if record["id"] in ids:
                record["access_counter"] += feedback
                record["important_score"] = self.importance_score_change_access_counter(
                    access_counter=record["access_counter"],
                    importance_score=record["important_score"],
                )
                record["important_score_recency_compound_score"] = self.compound_score_calculation_func.recency_and_importance_score(
                    recency_score=record["recency_score"],
                    importance_score=record["important_score"],
                )
                updated_ids.append(record["id"])

        # 점수가 변경되었으므로 리스트를 다시 정렬합니다.
        self.universe[symbol]["score_memory"]._rebuild()
        return updated_ids

    def _decay(self) -> None:
        """시간이 지남에 따라 모든 기억의 최신성 점수와 중요도 점수를 감소(decay)시킵니다."""
        for symbol in self.universe:
            for record in self.universe[symbol]["score_memory"]:
                record["recency_score"], record["important_score"], record["delta"] = self.decay_function(
                    important_score=record["important_score"], delta=record["delta"]
                )
                record["important_score_recency_compound_score"] = self.compound_score_calculation_func.recency_and_importance_score(
                    recency_score=record["recency_score"],
                    importance_score=record["important_score"],
                )
            self.universe[symbol]["score_memory"]._rebuild()

    def _clean_up(self) -> List[int]:
        """점수가 너무 낮아진(오래되고 중요하지 않은) 기억들을 데이터베이스에서 삭제합니다."""
        removed_ids_all = []
        for symbol in self.universe:
            rec_thresh = self.clean_up_threshold_dict["recency_threshold"]
            imp_thresh = self.clean_up_threshold_dict["importance_threshold"]

            to_remove = [r for r in self.universe[symbol]["score_memory"]
                         if r["recency_score"] < rec_thresh or r["important_score"] < imp_thresh]

            if to_remove:
                removed_ids = [r["id"] for r in to_remove]
                self.universe[symbol]["score_memory"] -= to_remove
                self.universe[symbol]["index"].remove_ids(np.array(removed_ids, dtype=np.int64))
                removed_ids_all.extend(removed_ids)
        return removed_ids_all

    def step(self) -> List[int]:
        """하루가 지날 때 수행하는 작업: 점수 감소 및 불필요한 기억 삭제"""
        self._decay()
        return self._clean_up()

    def prepare_jump(self) -> Tuple[Dict, Dict, List[int]]:
        """다른 기억 계층으로 이동('jump')할 기억들을 찾아 준비합니다."""
        jump_up = {}
        jump_down = {}
        ids_to_remove_from_db = []

        for symbol in self.universe:
            to_jump_up = [r for r in self.universe[symbol]["score_memory"] if r["important_score"] >= self.jump_threshold_upper]
            to_jump_down = [r for r in self.universe[symbol]["score_memory"] if r["important_score"] < self.jump_threshold_lower]

            if to_jump_up:
                ids = [r["id"] for r in to_jump_up]
                vectors = np.vstack([self.universe[symbol]["index"].reconstruct(id_) for id_ in ids])
                jump_up[symbol] = {"jump_object_list": to_jump_up, "emb_list": vectors}
                ids_to_remove_from_db.extend(ids)

            if to_jump_down:
                ids = [r["id"] for r in to_jump_down]
                vectors = np.vstack([self.universe[symbol]["index"].reconstruct(id_) for id_ in ids])
                jump_down[symbol] = {"jump_object_list": to_jump_down, "emb_list": vectors}
                ids_to_remove_from_db.extend(ids)

            # 현재 DB에서는 이동할 기억들을 삭제
            if ids_to_remove_from_db:
                 all_to_remove = to_jump_up + to_jump_down
                 self.universe[symbol]["score_memory"] -= all_to_remove
                 self.universe[symbol]["index"].remove_ids(np.array([r["id"] for r in all_to_remove], dtype=np.int64))

        return jump_up, jump_down, ids_to_remove_from_db

    def accept_jump(self, jump_info: Tuple[Dict, Dict], direction: str) -> None:
        """다른 기억 계층으로부터 이동해 온 기억들을 자신의 데이터베이스에 추가합니다."""
        jump_dict = jump_info[0] if direction == "up" else jump_info[1]

        for symbol, data in jump_dict.items():
            if symbol not in self.universe:
                self.add_new_symbol(symbol)

            records = data["jump_object_list"]
            vectors = data["emb_list"]
            ids = [r["id"] for r in records]

            # 상위 계층으로 이동하는 경우, 최신성 점수를 초기화
            if direction == "up":
                for r in records:
                    r["recency_score"] = self.recency_score_initialization_func()
                    r["delta"] = 0

            self.universe[symbol]["score_memory"].update(records)
            self.universe[symbol]["index"].add_with_ids(vectors, np.array(ids, dtype=np.int64))

    def save_checkpoint(self, name: str, path: str, force: bool = False):
        # ... (생략) ...
        pass

    @classmethod
    def load_checkpoint(cls, path: str) -> "MemoryDB":
        # ... (생략) ...
        pass

class BrainDB:
    """
    에이전트의 전체 기억 시스템('뇌')을 관리하는 최상위 클래스입니다.
    단기, 중기, 장기, 성찰 기억의 네 가지 계층을 총괄하며,
    기억의 추가, 검색, 업데이트 및 계층 간 이동(jump)을 관리합니다.
    """
    def __init__(
        self,
        agent_name: str,
        emb_config: Dict[str, Any],
        id_generator: id_generator_func,
        short_term_memory: MemoryDB,
        mid_term_memory: MemoryDB,
        long_term_memory: MemoryDB,
        reflection_memory: MemoryDB,
        logger: logging.Logger,
        use_gpu: bool = True,
    ):
        self.agent_name = agent_name
        self.emb_config = emb_config
        self.use_gpu = use_gpu
        self.id_generator = id_generator
        self.logger = logger
        # --- 4개의 기억 계층 ---
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.reflection_memory = reflection_memory
        self.removed_ids = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BrainDB":
        """설정 파일로부터 전체 기억 시스템(BrainDB)을 생성합니다."""
        id_generator = id_generator_func()
        agent_name = config["general"]["agent_name"]
        logger = logging.getLogger(__name__) # ... 로거 설정 생략 ...
        emb_config = config["agent"]["agent_1"]["embedding"]["detail"]

        # 각 기억 계층(MemoryDB)을 설정 파일에 따라 개별적으로 생성
        short_term_memory = MemoryDB(db_name="short", id_generator=id_generator, logger=logger, emb_config=emb_config, **config["short"])
        mid_term_memory = MemoryDB(db_name="mid", id_generator=id_generator, logger=logger, emb_config=emb_config, **config["mid"])
        long_term_memory = MemoryDB(db_name="long", id_generator=id_generator, logger=logger, emb_config=emb_config, **config["long"])
        reflection_memory = MemoryDB(db_name="reflection", id_generator=id_generator, logger=logger, emb_config=emb_config, **config["reflection"])

        return cls(
            agent_name=agent_name, id_generator=id_generator, logger=logger, emb_config=emb_config,
            short_term_memory=short_term_memory, mid_term_memory=mid_term_memory,
            long_term_memory=long_term_memory, reflection_memory=reflection_memory
        )

    # --- 각 기억 계층에 대한 추가/검색 메서드 ---
    def add_memory_short(self, symbol: str, date: date, text: Union[List[str], str]):
        self.short_term_memory.add_memory(symbol, date, text)
    # ... (add_memory_mid, add_memory_long, add_memory_reflection 생략) ...

    def query_short(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        return self.short_term_memory.query(query_text, top_k, symbol)
    # ... (query_mid, query_long, query_reflection 생략) ...

    def update_access_count_with_feed_back(self, symbol: str, ids: Union[List[int], int], feedback: int):
        """피드백을 모든 기억 계층에 전파하여 관련 기억의 점수를 업데이트합니다."""
        if isinstance(ids, int): ids = [ids]
        ids = [i for i in ids if i not in self.removed_ids]

        # 각 메모리 계층을 순회하며 ID에 해당하는 기억을 찾아 점수를 업데이트
        for memory_layer in [self.short_term_memory, self.mid_term_memory, self.long_term_memory, self.reflection_memory]:
            updated = memory_layer.update_access_count_with_feed_back(symbol, ids, feedback)
            ids = [i for i in ids if i not in updated] # 이미 업데이트된 ID는 제외
            if not ids: break

    def step(self):
        """하루가 지날 때 수행하는 작업: 점수 감소, 기억 삭제, 계층 간 이동"""
        # 1. 각 계층의 step()을 호출하여 점수 감소 및 오래된 기억 삭제
        for layer in [self.short_term_memory, self.mid_term_memory, self.long_term_memory, self.reflection_memory]:
            removed = layer.step()
            self.removed_ids.extend(removed)

        # 2. 기억 계층 간 이동(jump) 처리
        # 중요해진 단기기억 -> 중기기억으로 이동
        jump_info_short = self.short_term_memory.prepare_jump()
        self.mid_term_memory.accept_jump(jump_info_short, "up")

        # 중요해진 중기기억 -> 장기기억으로, 덜 중요해진 중기기억 -> 단기기억으로
        jump_info_mid = self.mid_term_memory.prepare_jump()
        self.long_term_memory.accept_jump(jump_info_mid, "up")
        self.short_term_memory.accept_jump(jump_info_mid, "down")

        # 덜 중요해진 장기기억 -> 중기기억으로
        jump_info_long = self.long_term_memory.prepare_jump()
        self.mid_term_memory.accept_jump(jump_info_long, "down")

    def save_checkpoint(self, path: str, force: bool = False):
        # ... (생략) ...
        pass

    @classmethod
    def load_checkpoint(cls, path: str):
        # ... (생략) ...
        pass

# 필요한 라이브러리들을 가져옵니다.
import os
import httpx  # HTTP 요청을 보내기 위한 라이브러리 (API 호출에 사용)
import json
import subprocess
from abc import ABC
from typing import Callable, Any, Dict, List

# --- TGI 모델용 프롬프트 빌더 ---

def build_llama2_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Llama2와 같은 특정 모델 형식에 맞게 대화 내용을 프롬프트 문자열로 변환합니다.
    모델이 지시사항을 잘 따르도록 특별한 태그([INST], <<SYS>>)를 사용합니다.
    """
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        # 첫 번째 시스템 메시지는 특별한 형식으로 감싸줍니다.
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        # 사용자의 메시지를 추가합니다.
        elif message["role"] == "user":
            conversation.append(message['content'].strip())
        # 이전 어시스턴트의 답변과 새로운 지시를 구분합니다.
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt


# --- 사용자 정의 예외 ---

class LongerThanContextError(Exception):
    """LLM에 보낸 텍스트가 모델이 처리할 수 있는 최대 길이(Context)를 초과했을 때 발생하는 예외입니다."""
    pass

# --- LLM API 호환 클래스 ---

class ChatOpenAICompatible(ABC):
    """
    OpenAI의 채팅 API와 유사한 인터페이스를 제공하여,
    GPT, Gemini, TGI(Text Generation Inference) 등 다양한 LLM 모델을
    일관된 방식으로 호출할 수 있도록 돕는 클래스입니다.
    """
    def __init__(
        self,
        end_point: str,  # API 요청을 보낼 주소 (URL)
        model: str = "gemini-pro",  # 사용할 LLM 모델의 이름
        system_message: str = "You are a helpful assistant.",  # LLM의 역할을 정의하는 시스템 메시지
        other_parameters: Dict[str, Any] | None = None,  # 모델별 추가 파라미터 (예: temperature)
    ):
        self.end_point = end_point
        self.model = model
        self.system_message = system_message
        self.other_parameters = {} if other_parameters is None else other_parameters

        # 모델 종류에 따라 HTTP 요청 헤더를 다르게 설정합니다.
        # 인증 방식이나 필요한 정보가 모델마다 다르기 때문입니다.
        if self.model.startswith("gemini-pro"):
            # Gemini는 gcloud 인증 토큰을 사용합니다.
            proc_result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
            access_token = proc_result.stdout.strip()
            self.headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        elif self.model.startswith("tgi"):
            # TGI는 별도의 인증 토큰이 필요 없을 수 있습니다.
            self.headers = {'Content-Type': 'application/json'}
        else:  # 기본값은 OpenAI 방식
            api_key = os.environ.get("OPENAI_API_KEY", "-")
            self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def parse_response(self, response: httpx.Response) -> str:
        """
        다양한 LLM API로부터 받은 응답(response)을 파싱하여
        순수한 텍스트 결과물만 추출합니다.
        """
        response_json = response.json()
        try:
            if self.model.startswith("gpt"):
                return response_json["choices"][0]["message"]["content"]
            elif self.model.startswith("gemini-pro"):
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            elif self.model.startswith("tgi"):
                return response_json["generated_text"]
            else:
                # 지원하지 않는 모델일 경우 오류를 발생시킵니다.
                raise NotImplementedError(f"Model {self.model} not implemented")
        except (KeyError, IndexError) as e:
            # 예상치 못한 응답 형식이 올 경우를 대비한 오류 처리
            raise ValueError(f"Error parsing LLM response: {response_json}") from e

    def guardrail_endpoint(self) -> Callable:
        """
        Guardrails 라이브러리와 함께 사용하기 위한 API 호출 함수(클로저)를 생성하여 반환합니다.
        이 함수는 Guardrails가 요구하는 간단한 (input: str) -> str 형식의 인터페이스를 제공합니다.
        """
        def end_point_func(input_str_from_guardrail: str, **kwargs) -> str:
            # Guardrails로부터 받은 문자열 입력을 LLM이 이해하는 대화 형식으로 변환합니다.
            # 시스템 메시지를 통해 LLM이 오직 JSON 형식으로만 답변하도록 유도합니다.
            messages = [
                {"role": "system", "content": "You are a helpful assistant only capable of communicating with valid JSON, and no other text."},
                {"role": "user", "content": input_str_from_guardrail},
            ]
            
            # 모델 종류에 따라 요청 페이로드(payload)를 다르게 구성합니다.
            if self.model.startswith("gemini-pro"):
                # Gemini 형식의 페이로드
                payload = {
                    "contents": [{"role": "USER", "parts": {"text": m["content"]}} for m in messages if m["role"] == "user"],
                    "generation_config": {"temperature": 0.2, "top_p": 0.1, ...}, # 기타 설정
                    "safety_settings": {...}
                }
            elif self.model.startswith("tgi"):
                # TGI (Llama2) 형식의 페이로드
                llama_input_str = build_llama2_prompt(messages)
                payload = {
                    "inputs": llama_input_str,
                    "parameters": {"max_new_tokens": 256, ...} # 기타 설정
                }
            else: # OpenAI 형식의 페이로드
                payload = {"model": self.model, "messages": messages}
                payload.update(self.other_parameters)

            # 구성된 페이로드를 사용하여 실제 HTTP POST 요청을 보냅니다.
            response = httpx.post(self.end_point, headers=self.headers, json=payload, timeout=600.0)

            try:
                # HTTP 오류 (예: 404, 500)가 발생하면 예외를 발생시킵니다.
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # 특히, 입력 텍스트가 너무 길어서 발생하는 오류(422)는 별도로 처리합니다.
                if response.status_code == 422 and "must have less than" in response.text:
                    raise LongerThanContextError from e
                else:
                    raise e

            # 성공적인 응답을 파싱하여 텍스트를 반환합니다.
            return self.parse_response(response)

        return end_point_func

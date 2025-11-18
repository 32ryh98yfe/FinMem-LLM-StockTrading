# 이 파일은 LLM(거대 언어 모델)에게 보낼 다양한 프롬프트(지시문) 템플릿을 정의합니다.
# 각 변수는 특정 상황에서 LLM의 역할이나 응답 형식을 지정하는 데 사용됩니다.

# --- LLM의 JSON 출력에서 특정 필드에 대한 설명 ---

# memory_ids: LLM이 어떤 기억을 참고했는지 ID를 출력하도록 유도하는 설명
short_memory_id_desc = "단기 기억 정보의 ID입니다."
mid_memory_id_desc = "중기 기억 정보의 ID입니다."
long_memory_id_desc = "장기 기억 정보의 ID입니다."
reflection_memory_id_desc = "성찰 기억 정보의 ID입니다."

# memory_id_extract_prompt: LLM이 특정 기억 계층에서 가장 관련 있는 정보를 선택하도록 지시하는 프롬프트
train_memory_id_extract_prompt = "제공된 정보({memory_layer}) 중에서 ARK, Two Sigma, Bridgewater Associates와 같은 주요 투자 회사의 투자 제안과 가장 관련이 있는 정보를 선택해주세요."
test_memory_id_extract_prompt = "제공된 정보({memory_layer}) 중에서 당신의 투자 결정과 가장 관련이 있는 정보를 선택해주세요."

# --- 거래 결정 요약에 대한 설명 ---

# trade_reason_summary: LLM이 투자 결정의 이유를 요약하도록 지시하는 프롬프트
train_trade_reason_summary = "전문 트레이더의 거래 제안이 주어졌을 때, 제공된 정보를 바탕으로 트레이더가 왜 그런 결정을 내렸는지 설명해주세요."
test_trade_reason_summary = "제공된 텍스트 정보와 주가 움직임 요약을 바탕으로, 왜 그런 투자 결정을 내렸는지 이유를 설명해주세요."
test_invest_action_choice = "제공된 정보를 바탕으로 투자 결정을 내려주세요: 주식 매수(buy), 매도(sell), 또는 보유(hold)."

# --- 프롬프트에 주입될 정보의 접두사 ---

# investment_info_prefix: 실제 시장 데이터(날짜, 종목, 가격 변동 등) 앞에 붙는 설명 템플릿
train_investment_info_prefix = "현재 날짜는 {cur_date}입니다. 관찰된 금융 시장 사실은 다음과 같습니다: {symbol} 종목의 경우, 다음 거래일과 현재 거래일 사이의 가격 차이는 {future_record}입니다.\n\n"
test_investment_info_prefix = "분석할 주식의 티커는 {symbol}이며 현재 날짜는 {cur_date}입니다."

# sentiment_explanation & momentum_explanation: 테스트 모드에서 감성 점수와 모멘텀 정보의 의미를 LLM에게 설명
test_sentiment_explanation = """
예를 들어, 회사에 대한 긍정적인 뉴스는 투자 심리를 개선하여 더 많은 매수 활동을 유도하고 주가를 상승시킬 수 있습니다.
반대로, 부정적인 뉴스는 투자 심리를 악화시켜 매도 압력을 유발하고 주가 하락으로 이어질 수 있습니다.
경쟁사에 대한 뉴스 또한 회사의 주가에 파급 효과를 미칠 수 있습니다.
긍정 점수, 중립 점수, 부정 점수는 감성 점수를 나타내며, 각 텍스트가 해당 카테고리에 속하는 비율을 의미합니다 (총합은 1).
"""
test_momentum_explanation = """
아래 정보는 주식의 '모멘텀'을 반영하는 지난 며칠간의 주가 변동 요약입니다.
모멘텀은 과거에 좋은 성과를 보인 증권이 계속해서 좋은 성과를 낼 것이라는 아이디어를 기반으로 합니다.
"""

# --- 전체 프롬프트 템플릿 ---

# train_prompt: '학습(Train)' 모드에서 사용할 전체 프롬프트.
# LLM에게 미래의 주가 변동(정답)을 알려주고, 그 이유를 제공된 정보를 바탕으로 설명하도록 요청합니다.
train_prompt = """주어진 정보를 바탕으로, 현재 날짜부터 다음 날까지의 금융 시장 변동이 왜 그렇게 발생했는지 설명해주시겠습니까? 결정의 이유를 요약해주세요.
    요약 정보와 그 요약을 뒷받침하는 정보의 ID를 제공해야 합니다.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
    당신의 출력은 다른 추가 내용 없이 다음 JSON 형식을 엄격하게 따라야 합니다: {"summary_reason": string, "short_memory_index": number, "middle_memory_index": number, "long_memory_index": number, "reflection_memory_index": number}
"""

# test_prompt: '테스트(Test)' 모드에서 사용할 전체 프롬프트.
# LLM에게 다양한 정보를 제공하고, 이를 종합하여 '매수', '매도', '보유' 중 하나의 투자 결정을 내리고 그 이유를 설명하도록 요청합니다.
test_prompt = """주어진 정보를 바탕으로 투자 결정을 내려주시겠습니까? 결정의 이유를 요약해주세요.
    사용 가능한 단기, 중기, 장기, 성찰 정보만을 고려해야 합니다.
    과거 주가의 모멘텀을 고려해야 합니다.
    누적 수익률이 양수이거나 0일 경우, 당신은 위험 추구형 투자자입니다.
    투자자가 현재 얼마나 많은 주식을 보유하고 있는지도 고려해야 합니다.
    반드시 다음 투자 결정 중 하나를 제공해야 합니다: 'buy' 또는 'sell'.
    'buy' 또는 'sell' 결정을 내리기 정말 어려운 경우에만 'hold' 옵션을 선택할 수 있습니다.
    또한 당신의 결정을 뒷받침하는 정보의 ID를 제공해야 합니다.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
"""

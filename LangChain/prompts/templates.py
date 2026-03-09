from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_template(
    """다음 문맥을 참고하여 질문에 답변해주세요.
문맥에서 답을 찾을 수 없다면 모른다고 답변해주세요.

문맥:
{context}

질문: {question}

답변:"""
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """너는 검색 쿼리를 확장하는 AI다.
사용자의 질문에 등장하는 용어를 한국어·영어·약어·전체 표현으로 확장하여
검색에 최적화된 단일 쿼리 문자열로 반환한다.

규칙:
- 한국어 용어 → 영어 번역 추가 (예: 표 1 → Table 1)
- 영어 용어 → 한국어 번역 추가 (예: attention → 어텐션)
- 약어 → 전체 표현 추가 (예: FFN → Feed Forward Network)
- 전체 표현 → 약어 추가 (예: Feed Forward Network → FFN)
- 표/그림 번호는 한국어·영어·공백 유무 모두 포함 (예: 표1, 표 1, 테이블1, 테이블 1, Table1, Table 1)
- 원본 질문의 의미는 유지
- 추가 설명 없이 확장된 쿼리 문자열만 출력"""
    ),
    (
        "human",
        "다음 질문을 확장하라.\n\n질문: {question}"
    )
])
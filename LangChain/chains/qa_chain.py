import re
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from prompts.templates import QA_PROMPT, REWRITE_PROMPT
from config import Config

config = Config()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_table_numbers(text: str) -> list:
    """쿼리에서 표 번호 추출 (표1, 테이블 2, Table 3 등)"""
    patterns = [r'표\s*(\d+)', r'테이블\s*(\d+)', r'[Tt]able\s*(\d+)']
    numbers = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            numbers.add(int(match.group(1)))
    return list(numbers)


def make_retriever_fn(vectorstore, table_registry: dict):
    """
    표 번호가 언급된 경우 table_registry에서 직접 조회하여 컨텍스트에 강제 포함.
    벡터 검색의 시맨틱 유사도 문제를 우회.
    """
    def retrieve(query: str) -> list:
        # 일반 시맨틱 검색
        docs = vectorstore.similarity_search(query, k=6)
        seen = {d.page_content for d in docs}

        # 표 번호가 언급된 경우 레지스트리에서 직접 조회
        for num in extract_table_numbers(query):
            if num in table_registry:
                d = table_registry[num]
                if d.page_content not in seen:
                    docs.append(d)
                    seen.add(d.page_content)

        return docs

    return retrieve


def create_qa_chain(vectorstore, table_registry: dict = None):
    llm = ChatOpenAI(model_name=config.LLM_MODEL, temperature=0)
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    retrieve_fn = make_retriever_fn(vectorstore, table_registry or {})

    qa_chain = (
        {
            "context": (
                RunnablePassthrough()
                | RunnableLambda(lambda q: rewrite_chain.invoke({"question": q}))
                | RunnableLambda(retrieve_fn)
                | format_docs
            ),
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain

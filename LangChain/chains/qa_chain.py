from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from prompts.templates import QA_PROMPT, REWRITE_PROMPT
from config import Config

config = Config()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name=config.LLM_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 쿼리 재작성 체인: 질문 → LLM이 한/영 + 약어 확장 → 벡터 검색
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    qa_chain = (
        {
            "context": (
                RunnablePassthrough()
                | RunnableLambda(lambda q: rewrite_chain.invoke({"question": q}))
                | retriever
                | format_docs
            ),
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain

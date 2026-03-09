from loaders.loader import load_pdf, split_documents
from utils.vectorstore import create_vectorstore
from chains.qa_chain import create_qa_chain


def main():
    # 1. PDF 로드 및 분할
    pdf_path = "./sample/sample.pdf"
    documents, table_registry = load_pdf(pdf_path)

    chunks = split_documents(documents)

    # 2. 벡터스토어 생성
    vectorstore = create_vectorstore(chunks)

    # 3. QA 체인 생성 (표 직접 조회용 레지스트리 함께 전달)
    qa_chain = create_qa_chain(vectorstore, table_registry)

    # 4. 질의응답 루프
    print("PDF 챗봇이 준비되었습니다. 종료하려면 'quit'을 입력하세요.")
    while True:
        question = input("\n질문: ")
        if question.lower() == "quit":
            break
        result = qa_chain.invoke(question)
        print(f"\n답변: {result}")


if __name__ == "__main__":
    main()
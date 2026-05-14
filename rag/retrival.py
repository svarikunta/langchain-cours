import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

embeddings = OpenAIEmbeddings()
llm =ChatOpenAI(model="gpt-5.2")

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriver= vectorstore.as_retriever(search_kwars={"k: 3"})

prompt_template = ChatPromptTemplate.from_template("""Answer the question based only on the following context :
{context}

question : {question}

provide a detailed answer:
""")


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    simple retrieval chain without LCEL
    manually retrieves documents, formats them, and generate a response.
    :param query:
    :return:
    """
    docs=retriver.invoke(query)
    context =format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)
    return response.content


def retrieval_chain_with_lcel():
    """

    :return:
    """
    retrieval_chain =(
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriver | format_docs
        )
        | prompt_template | llm | StrOutputParser()

    )

    return retrieval_chain



def main():
    # print("vector db retrival!")
    #
    query = "What is Pinecone in machine leaning?"
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No Rag)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer")
    # print(result_raw.content)
    # result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    # print(result)


    print("\n" + "with LCEL" + "=" * 70)
    result=retrieval_chain_with_lcel().invoke({"question": query})
    print("\nAnswer")
    print(result)


if __name__ == "__main__":
    main()



from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

search = DuckDuckGoSearchRun()
spiltter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap=200
)

embeddings = HuggingFaceEmbeddings(
    model="all-MiniLM-L6-v2"
)



@tool
def web_search(user_query:str)->str:
    """
    Search the web for the real time inforamtion based on user search query.
    """
    response = search.invoke(user_query)
    return response

@tool
def digital_marketing_rag(user_query:str)->str:
    """
    Search the document for the relvelent anwers for user question
    """
    loader = PyPDFLoader("data/smm.pdf")
    doc = loader.load()
    print(f"PDF loaded")

    chunks = spiltter.split_documents(doc)
    print(f"Total {len(chunks)} created for file.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k":4}
    )

    retrieved_chunks = retriever.invoke(user_query)
    context = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)
    return context if context else "No relevent information found"


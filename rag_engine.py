import os
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


PDF_PATH ="constitution_of_india.pdf"
PERSIST_DIR = "Constitution_Rag"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
LAMBDA_MULT = 0.8

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


PROMPT_TEMPLATE = """
You are a professional AI Legal Assistant specialised in the Constitution of India.

You are provided with excerpts retrieved from a vector database built on the Constitution's text.
These excerpts are your ONLY source of information.

Instructions:
- Read all retrieved excerpts carefully before answering.
- Base every statement strictly on the provided constitutional text.
- Never fabricate or infer missing information.
- Cite the relevant Article number(s) whenever available.
- If the answer cannot be determined from the retrieved excerpts, respond exactly:

"I couldn't find sufficient information in the retrieved excerpts."

Response Guidelines:
- Mention the Article number and title whenever available.
- Summarize information rather than copying long passages verbatim.
- Use bullet points when multiple clauses or articles are relevant.
- Keep answers factual and concise.
- If confidence is low because the retrieved context is incomplete, explicitly state that.

Retrieved Constitutional Context
========================
{context_text}
========================

Question:
{user_query}

Grounded Answer:
"""


def check_api_key():
    """Returns True if the OpenAI API key is set."""
    return bool(OPENAI_API_KEY)


def check_pdf_exists():
    """Returns PDF diagnostics."""
    exists = os.path.exists(PDF_PATH)
    cwd = os.getcwd()

    files_in_data = (
        os.listdir("data")
        if os.path.isdir("data")
        else None
    )

    return exists, cwd, files_in_data


def create_vector_store():
    """Create a new Chroma vector store from the Constitution PDF."""

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    vector_store = Chroma(
        collection_name="constitution",
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )

    vector_store.add_documents(chunks)

    return vector_store, len(chunks)


def build_or_load_retriever():
    """
    Load existing Chroma index.
    If no documents exist, create the index from PDF.

    Returns:
        retriever, number_of_chunks
    """

    embedding = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    vector_store = Chroma(
        collection_name="constitution",
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )

    existing_count = vector_store._collection.count()

    if existing_count == 0:

        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(docs)

        vector_store.add_documents(chunks)

        num_chunks = len(chunks)

    else:
        num_chunks = existing_count

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "lambda_mult": LAMBDA_MULT
        }
    )

    return retriever, num_chunks


def rebuild_index():
    """Delete existing index and rebuild it from the PDF."""

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    return build_or_load_retriever()


def get_llm():
    """Return the chat LLM."""

    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )


def answer_question(retriever, llm, user_query: str):
    """
    Retrieve relevant constitutional passages
    and generate a grounded answer.
    """

    retrieved_docs = retriever.invoke(user_query)

    if not retrieved_docs:
        return (
            "I couldn't find sufficient information in the retrieved excerpts.",
            []
        )

    context_text = "\n\n".join(
        f"""
[source: {doc.metadata.get('source')}
page: {doc.metadata.get('page')}]

{doc.page_content}
"""
        for doc in retrieved_docs
    )

    prompt = PROMPT_TEMPLATE.format(
        context_text=context_text,
        user_query=user_query
    )

    try:
        result = llm.invoke(prompt)
        answer = result.content

    except Exception as e:
        answer = f" Error generating answer: {e}"

    return answer, retrieved_docs
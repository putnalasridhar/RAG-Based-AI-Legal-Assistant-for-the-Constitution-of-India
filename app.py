"""
app.py
=========================================================
Streamlit INTERFACE only.

All RAG logic is handled by rag_engine.py.
=========================================================
"""

import streamlit as st
import rag_engine as rag


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Constitution of India - RAG Legal Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>

    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    .app-header {
        background: linear-gradient(
            135deg,
            #1e3a5f 0%,
            #2c5282 100%
        );
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        color: white;
    }

    .app-header h1 {
        margin: 0;
        font-size: 1.7rem;
    }

    .app-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .status-ready {
        background: #d4edda;
        color: #155724;
    }

    .source-chip {
        display: inline-block;
        background: #eef2f7;
        color: #2c5282;
        padding: 0.15rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        margin-right: 0.4rem;
    }

    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title("Constitution of India — RAG Legal Assistant")

st.caption(
    "Grounded Q&A over the Constitution of India — just ask, no setup needed."
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# API KEY CHECK
# =========================================================

if not rag.check_api_key():
    st.error(
        """
        OPENAI_API_KEY not found.

        Add your API key to the `.env` file:

        OPENAI_API_KEY=sk-...
        """
    )
    st.stop()


# =========================================================
# PDF CHECK
# =========================================================

pdf_exists, cwd, files_in_data = rag.check_pdf_exists()

if not pdf_exists:
    st.error(
        f"""
        Constitution PDF not found.

        Expected PDF:

        `{rag.PDF_PATH}`

        Current working directory:

        `{cwd}`

        Make sure `constitution_of_india.pdf`
        is available in the project folder.
        """
    )
    st.stop()


# =========================================================
# LOAD RETRIEVER
# =========================================================

@st.cache_resource(show_spinner=False)
def load_retriever():
    return rag.build_or_load_retriever()


# =========================================================
# LOAD LLM
# =========================================================

@st.cache_resource(show_spinner=False)
def load_llm():
    return rag.get_llm()


# =========================================================
# INITIALIZE RAG
# =========================================================

with st.spinner("Preparing the Constitution index..."):
    retriever, num_chunks = load_retriever()

llm = load_llm()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # =====================================================
    # 1. STATUS
    # =====================================================

    st.header("Status")

    st.markdown(
        f"""
        <span class="status-pill status-ready">
             Index ready · {num_chunks} chunks
        </span>
        """,
        unsafe_allow_html=True
    )

    st.caption(f"Source: `{rag.PDF_PATH}`")

    st.caption(f"Vector store: `{rag.PERSIST_DIR}`")

    st.markdown("---")

    # =====================================================
    # 2. CHUNKING
    # =====================================================

    st.subheader("2. Chunking")

    # -----------------------------------------------------
    # CHUNK SIZE
    # -----------------------------------------------------

    chunk_size = st.slider(
        "Chunk size",
        min_value=100,
        max_value=2000,
        value=rag.CHUNK_SIZE,
        step=50,
        help=(
            "Number of characters used for "
            "each document chunk."
        )
    )

    # -----------------------------------------------------
    # CHUNK OVERLAP
    # -----------------------------------------------------

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=rag.CHUNK_OVERLAP,
        step=10,
        help=(
            "Number of characters shared "
            "between adjacent chunks."
        )
    )

    st.markdown("---")

    # =====================================================
    # 3. RETRIEVAL (MMR)
    # =====================================================

    st.subheader("3. Retrieval (MMR)")

    # -----------------------------------------------------
    # TOP K
    # -----------------------------------------------------

    top_k = st.slider(
        "Top K results",
        min_value=1,
        max_value=15,
        value=rag.TOP_K,
        step=1,
        help=(
            "Number of relevant chunks retrieved "
            "for each question."
        )
    )

    # -----------------------------------------------------
    # LAMBDA
    # -----------------------------------------------------

    lambda_mult = st.slider(
        "Lambda (diversity vs relevance)",
        min_value=0.0,
        max_value=1.0,
        value=float(rag.LAMBDA_MULT),
        step=0.05,
        help=(
            "Higher values favor relevance. "
            "Lower values favor diversity."
        )
    )

    st.markdown("---")

    # =====================================================
    # CHECK SETTINGS
    # =====================================================

    settings_changed = (
        chunk_size != rag.CHUNK_SIZE
        or
        chunk_overlap != rag.CHUNK_OVERLAP
        or
        top_k != rag.TOP_K
        or
        lambda_mult != rag.LAMBDA_MULT
    )

    # =====================================================
    # APPLY & REBUILD
    # =====================================================

    if settings_changed:

        st.warning("Settings changed.")

        if st.button(
            "Apply & Rebuild Index",
            use_container_width=True
        ):
            try:
                with st.spinner("Rebuilding Constitution index..."):

                    # -------------------------------------
                    # Update configuration
                    # -------------------------------------

                    rag.CHUNK_SIZE = chunk_size
                    rag.CHUNK_OVERLAP = chunk_overlap
                    rag.TOP_K = top_k
                    rag.LAMBDA_MULT = lambda_mult

                    # -------------------------------------
                    # Clear Streamlit cache
                    # -------------------------------------

                    st.cache_resource.clear()

                    # -------------------------------------
                    # Rebuild Chroma index
                    # -------------------------------------

                    retriever, num_chunks = rag.rebuild_index()

                st.success(
                    f"Index rebuilt successfully — "
                    f"{num_chunks} chunks."
                )

                # -----------------------------------------
                # Reload application
                # -----------------------------------------

                st.rerun()

            except Exception as e:
                st.error(f"Error rebuilding index: {e}")

    else:
        st.caption("Current settings are active.")

    st.markdown("---")

    # =====================================================
    # FORCE REBUILD
    # =====================================================

    if st.button(
        "Force rebuild index",
        use_container_width=True
    ):
        try:
            with st.spinner("Rebuilding Constitution index..."):
                # Rebuild with current settings
                retriever, num_chunks = rag.rebuild_index()

            st.success(
                f"Index rebuilt successfully — "
                f"{num_chunks} chunks."
            )

            st.rerun()

        except Exception as e:
            st.error(f"Error rebuilding index: {e}")

    st.markdown("---")

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    if st.session_state.messages:

        chat_text = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages
        )

        # -------------------------------------------------
        # DOWNLOAD CHAT
        # -------------------------------------------------

        st.download_button(
            "⬇Download chat history",
            data=chat_text,
            file_name="constitution_qa_history.txt",
            use_container_width=True
        )

        # -------------------------------------------------
        # CLEAR CHAT
        # -------------------------------------------------

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# =========================================================
# MAIN CHAT AREA
# =========================================================

# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================

for msg in st.session_state.messages:
    
    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        # -------------------------------------------------
        # DISPLAY SOURCES
        # -------------------------------------------------

        if msg.get("sources"):

            with st.expander(
                f"{len(msg['sources'])} retrieved excerpt(s)"
            ):

                for doc in msg["sources"]:

                    # -----------------------------
                    # Source filename
                    # -----------------------------

                    src = str(
                        doc.metadata.get("source", "Unknown")
                    )

                    src = src.split("/")[-1].split("\\")[-1]

                    # -----------------------------
                    # Page
                    # -----------------------------

                    page = doc.metadata.get("page", "N/A")

                    # -----------------------------
                    # Source chips
                    # -----------------------------

                    st.markdown(
                        f"""
                        <span class="source-chip">
                            {src}
                        </span>

                        <span class="source-chip">
                            page {page}
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

                    # -----------------------------
                    # Retrieved text
                    # -----------------------------

                    st.code(
                        doc.page_content,
                        language="text"
                    )


# =========================================================
# CHAT INPUT
# =========================================================

user_query = st.chat_input(
    "Ask a question about the Constitution, "
    "e.g. What does Article 21 say?"
)


# =========================================================
# PROCESS USER QUESTION
# =========================================================

if user_query:

    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
            "sources": None
        }
    )

    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    with st.chat_message("user"):
        st.markdown(user_query)

    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving context and "
            "generating grounded answer..."
        ):
            answer, retrieved_docs = rag.answer_question(
                retriever,
                llm,
                user_query
            )

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        st.markdown(answer)

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        if retrieved_docs:

            with st.expander(
                f"{len(retrieved_docs)} "
                f"retrieved excerpt(s)"
            ):

                for doc in retrieved_docs:

                    # -----------------------------
                    # Source
                    # -----------------------------

                    src = str(doc.metadata.get("source", "Unknown"))
                    src = src.split("/")[-1].split("\\")[-1]

                    # -----------------------------
                    # Page
                    # -----------------------------

                    page = doc.metadata.get("page", "N/A")

                    # -----------------------------
                    # Source chips
                    # -----------------------------

                    st.markdown(
                        f"""
                        <span class="source-chip">
                             {src}
                        </span>

                        <span class="source-chip">
                            page {page}
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

                    # -----------------------------
                    # Retrieved text
                    # -----------------------------

                    st.code(doc.page_content, language="text")

    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_docs
        }
    )
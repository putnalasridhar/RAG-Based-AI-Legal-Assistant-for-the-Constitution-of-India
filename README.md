Constitution of India Using RAG

An AI-based Legal Assistant that uses RAG (Retrieval-Augmented Generation) and LLM technology to answer questions related to the Constitution of India.

Features
📚 Uses the Constitution of India as the knowledge source.
🔍 Uses MMR retrieval to retrieve relevant constitutional information.
🤖 Uses an LLM to generate grounded answers.
📊 Evaluated using Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
Data Processing
Source: Constitution of India PDF
Pages: 268
Chunks: 1,376
Chunk Size: 500
Chunk Overlap: 100
Retrieval: MMR
k = 5
lambda_mult = 0.8
RAG Pipeline
PDF → Text Extraction → Chunking → Embeddings
→ Vector Database → MMR Retrieval → LLM → Answer
Objective

To provide quick, relevant, and grounded constitutional information for students, citizens, teachers, and researchers.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
  
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",   # small and fast 
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.from_documents(kb_docs, embed_model)

# a function to extract top_4 vectors from db for a query using cosine similarity
def faiss_context(query, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join(d.page_content for d in docs)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = OllamaEmbedding(model_name="snowflake-arctic-embed:33m", base_url="http://localhost:11434",ollama_additional_kwargs={"mirostat": 0})

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
query = "get me an overview on III YEAR II SEMESTER syllabus?"
print(f"Question: {query}")
response = query_engine.query(query)
print(f"Answer: {response}")
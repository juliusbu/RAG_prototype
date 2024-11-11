from retrieval_augmented_generation.rag import RAG
from retrieval_augmented_generation.configs import TextGenerationConfig, RetrievalConfig

rag = RAG('../Dokumente')

retrieval_config = RetrievalConfig(
    embedding_model_name="intfloat/multilingual-e5-base",
    embedding_query_template="passage:{text}",
    retrieval_query_template="query:{question}"
)

text_generation_config = TextGenerationConfig(
    text_generation_model_name="google/gemma-1.1-2b-it",
)
rag.init_huggingface(
    hf_transformers_cache_dir="./../../hf_transformers_cache",
    hf_hub_api_key="hf_XXXX",
    retrieval_config=retrieval_config,
    text_generation_config=text_generation_config
)

rag.update_files()

while True:
    question = input("\nQuestion:\n")
    answer = rag.ask(question)

    print("Answer:")
    print(answer['answer'], "\n")
    


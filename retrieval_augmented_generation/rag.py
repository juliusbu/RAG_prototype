from .configs import RetrievalConfig, TextGenerationConfig

from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import pypdfium2 as pdfium
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from jinja2.exceptions import TemplateError
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

from typing import Union
import torch
import os
import json
import random
import gc
import time
import multiprocessing
import math



class RAG:
    """
    Class to perform RAG.
    """

    def __init__(
            self,
            files_directory: str,
    ) -> None:
        """
        :param files_directory: Directory in which the documents for the Retrieval are stored (currently PDF or TXT)
        """
        self.__own_directory = os.path.dirname(os.path.abspath(__file__))
        self.files_directory = files_directory
        self.__db_directory = None
        self.__retrieval_config = None
        self.__text_generation_config = None
        self.__hf_transformers_cache_dir = os.path.join(self.__own_directory, "/hf_transformers_cache")
        file_message_templates = open(os.path.join(self.__own_directory,"message_templates.json"), 'r')
        self.__message_templates = json.load(file_message_templates)
        file_message_templates.close()
        self.__is_hf = None
        self.__prefix_retrieval_query = ""
        self.__embedding = None
        self.__db = None
        self.__tokenizer = None
        self.__model = None
        self.__text_generation_pipeline = None
        self.__num_processed = multiprocessing.Value('i', 0)
        self.__embedding_template = "{text}"
        self.__retrieval_template = "{question}"

    def __del__(self):
        del self.__tokenizer
        del self.__model
        gc.collect()
        torch.cuda.empty_cache()

    def init_huggingface(
            self,
            hf_hub_api_key: str = None,
            retrieval_config: RetrievalConfig = RetrievalConfig(),
            text_generation_config: TextGenerationConfig = TextGenerationConfig(),
            embedding_model_name: str = None,
            text_generation_model_name: str = None,
            hf_transformers_cache_dir: str = None,
    ) -> None:
        """
        Method to use, before the class can be used for RAG with models from the Hugging Face Hub.

        :param hf_hub_api_key: Hugging Face Hub API Key - required for some models where access is restricted. Can be obtained here: https://huggingface.co/settings/tokens
        :param retrieval_config: Optional Parameter to configure Retrieval
        :param text_generation_config: Optional Parameter to configure Text Generation
        :param embedding_model_name: Name of the model used to Generate the embedding vectors. Required if retrieval_config is not used
        :param text_generation_model_name: Name of the model used to generate text. Required if text_generation_config is not used.
        :param hf_transformers_cache_dir: Directory in which files downloaded from the Hugging Face Hub should be / are stored.
        """
        self.__is_hf = True

        self.__retrieval_config = retrieval_config
        self.__text_generation_config = text_generation_config

        # define embedding model name from either retrieval_config parameter or embedding_model_name parameter:
        if embedding_model_name is None and retrieval_config.embedding_model_name is not None:
            embedding_model_name = retrieval_config.embedding_model_name
        if embedding_model_name is None:
            raise TypeError("No name for embedding model given")


        self.__db_directory = os.path.join(
            retrieval_config.db_directory,
            retrieval_config.collection_name,
            embedding_model_name
        )

        # define text generation name from eiter text_generation_config parameter or text_generation_model_name parameter:
        if text_generation_model_name is None and text_generation_config.text_generation_model_name is not None:
            text_generation_model_name = text_generation_config.text_generation_model_name
        if text_generation_model_name is None:
            raise TypeError("No name for text generation model given")
        if hf_transformers_cache_dir is not None:
            self.__hf_transformers_cache_dir = hf_transformers_cache_dir

        # define templates to use for embedding and retrieval:
        self.__embedding_template = retrieval_config.embedding_query_template
        self.__retrieval_template = retrieval_config.retrieval_query_template

        # initialize db to be used for RAG, which includes the embedding-model:
        self.__embedding = _Embedding(
            embedding_model_name,
            hf_transformers_cache_dir
        )
        self.__db = _DbAdapter(
            persist_directory=self.__db_directory,
            embedding_function=self.__embedding,
            n_retrievals=self.__retrieval_config.n_retrievals
        )

        # define quantization to load the model in lower precision, to save resources:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Download and load both models - either with or without API Key:
        if hf_hub_api_key is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                text_generation_model_name,
                token=hf_hub_api_key,
                cache_dir=self.__hf_transformers_cache_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                text_generation_model_name,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                token=hf_hub_api_key,
                cache_dir=self.__hf_transformers_cache_dir
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                text_generation_model_name,
                cache_dir=self.__hf_transformers_cache_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                text_generation_model_name,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                cache_dir=self.__hf_transformers_cache_dir
            )
        self.__tokenizer = tokenizer
        self.__model = model
        print("memory allocated by CUDA: ", torch.cuda.memory_allocated() / 1000000000, "GB")
        print("memory reserved by CUDA: ", torch.cuda.memory_reserved() / 1000000000, "GB")

        # define text generation pipeline, which takes input and generates output:
        text_generation_pipeline = pipeline(task="text-generation",
                                            tokenizer=tokenizer,
                                            model=model,
                                            
                                            # ---- Contrastive search:
                                            # penalty_alpha=0.6,
                                            # top_k=4,
                                            
                                            # ---- Beam-search decoding:
                                            num_beams=5,
                                            
                                            # ---- Beam-search multinomial sampling:
                                            # num_beams=3,
                                            # do_sample=True,
                                            
                                            max_new_tokens=500,
                                            framework="pt",
                                            return_full_text=False
                                            )
        self.__text_generation_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

    def init_openai(
            self,
            open_ai_key: str,
            retrieval_config: RetrievalConfig = RetrievalConfig(),
            text_generation_config: TextGenerationConfig = TextGenerationConfig(),
            embedding_model_name: str = None,
            text_generation_model_name: str = None
    ) -> None:
        """
        Method to use, before the class can be used for RAG with OpenAI Models.

        :param open_ai_key: OpenAI API key. Can be obtained here: https://platform.openai.com/api-keys
        :param retrieval_config: Optional Parameter to configure Retrieval
        :param text_generation_config: Optional Parameter to configure Text Generation
        :param embedding_model_name: Name of the model used to Generate the embedding vectors. Required if retrieval_config is not used
        :param text_generation_model_name: Name of the model used to generate text. Required if text_generation_config is not used.
        """
        self.__is_hf = False

        self.__retrieval_config = retrieval_config
        self.__text_generation_config = text_generation_config

        # define embedding model name from either retrieval_config parameter or embedding_model_name parameter:
        if embedding_model_name is None and retrieval_config.embedding_model_name is not None:
            embedding_model_name = retrieval_config.embedding_model_name
        if embedding_model_name is None:
            embedding_model_name = "text-embedding-ada-002"

        self.__db_directory = os.path.join(
            retrieval_config.db_directory,
            retrieval_config.collection_name,
            embedding_model_name
        )

        # define text generation name from eiter text_generation_config parameter or text_generation_model_name parameter:
        if text_generation_model_name is None and text_generation_config.text_generation_model_name is not None:
            text_generation_model_name = text_generation_config.text_generation_model_name
        if text_generation_model_name is None:
            text_generation_model_name = "gpt-3.5-turbo"

        os.environ["OPENAI_API_KEY"] = open_ai_key

        # initialize db to be used for RAG, which includes the embedding-model:
        self.__embedding = OpenAIEmbeddings(model=embedding_model_name)
        self.__db = _DbAdapter(
            persist_directory=self.__db_directory,
            embedding_function=self.__embedding,
            n_retrievals=self.__retrieval_config.n_retrievals
        )

        # define text generation pipeline, which takes input and generates output:
        self.__text_generation_pipeline = ChatOpenAI(model_name=text_generation_model_name, temperature=0)

    def _init_nomodel(
            self,
            retrieval_config: RetrievalConfig = RetrievalConfig()
    ):
        """
        Not intended to be used. For debugging purposes only.
        :param retrieval_config: Parameter to configure Retrieval
        """
        self.__db_directory = os.path.join(
            retrieval_config.db_directory,
            retrieval_config.collection_name,
            "nomodel"
        )

        # initialize db to be used for RAG, which includes the embedding-model:
        self.__embedding = _EmbeddingDummy()
        self.__db = _DbAdapter(
            persist_directory=self.__db_directory,
            embedding_function=self.__embedding,
            n_retrievals=1
        )

    def __retriever_adapter(self, question: str) -> list[Document]:
        retrieval_query = self.__apply_retrieval_template(question)
        return self.__db.retrieve(retrieval_query)

    def __apply_embedding_template(self, text: str) -> str:
        return self.__embedding_template.replace("{text}", text)

    def __apply_retrieval_template(self, question: str) -> str:
        return self.__retrieval_template.replace("{question}", question)

    def update_files(self):
        """
        Generates new vector embeddings for new and updated documents and stores them persistently.
        """
        if self.__db is None:
            raise Exception("No Models have been initialized yet. Use init_huggingface or init_openai first.")

        # create array of all documents:
        path_list = [self.files_directory]
        list_files = []
        while len(path_list) > 0:
            path = path_list.pop()
            for element in os.listdir(path):
                element_path = os.path.join(path, element)
                # if element is folder
                if os.path.isdir(element_path):
                    path_list.append(element_path)
                else:
                    # if there is a function to read element with its file extension:
                    list_files.append({'file_path': element_path, 'mod_date_file': os.path.getmtime(element_path)})

        print(
            time.strftime("%H:%M:%S", time.localtime(time.time())),
            " -  processing", len(list_files), "files"
        )

        # remove files that don't have to be updated:
        def is_file_new(file):
            path = file['file_path']
            metadatas = self.__db.get_id(path + ".0.0")['metadatas']
            if len(metadatas) == 0:
                mod_date_embedding = None
            elif not metadatas[0]['mod_date']:
                mod_date_embedding = None
            else:
                mod_date_embedding = float(metadatas[0]['mod_date'])
            return (mod_date_embedding is None or (float(mod_date_embedding) < float(file['mod_date_file'])))
        list_files_new = list(filter(is_file_new, list_files))
        print(time.strftime("%H:%M:%S", time.localtime(time.time())), " - ", len(list_files_new), "files have to be updated")

        # update all files that have to be updated in chunks:
        chunk_size = 5000
        nr_of_chunks = math.ceil(len(list_files_new) / chunk_size)
        progress_bar_chunk = tqdm(total=nr_of_chunks)
        for i_chunk in range(0, len(list_files_new), chunk_size):
            files_chunk = list_files_new[i_chunk:i_chunk + chunk_size]
            self.__update_files_chunk(files_chunk)
            progress_bar_chunk.update(1)
        progress_bar_chunk.close()

        print("all tasks completed")

    def __update_files_chunk(self, files):

        # read content of files:
        new_files = []
        for file in files:
            new_files.append(_ReadFile.read(file))
        files = new_files

        # create vectors while storing in database:
        for file in files:
            texts = []
            metadatas = []
            ids = []
            embeddings=[]
            # if file is of non-compatible format, there won't be any text
            if "docs" in file:
                path = file['file_path']
                docs = file['docs']
                mod_date = file['mod_date_file']
                for i, page in enumerate(docs):
                    for j, text_fragment in enumerate(page):
                        texts.append(text_fragment.page_content)
                        metadatas.append({"source": path, "page": i, "chunk": j, "mod_date": str(mod_date)})
                        ids.append((path + "." + str(i) + "." + str(j)))
                        embeddings.append(self.__embedding.embed_query(self.__apply_embedding_template(text_fragment.page_content)))
                try:
                    self.__db.add_file(texts = texts, metadatas = metadatas, ids = ids, embeddings = embeddings)
                    print("file written")
                except ValueError:
                    pass

    def __prepare_rag_chain(self, message_template_name: str, context: list[Document]):

        llm = self.__text_generation_pipeline

        # function to build a string out of retrieved documents:
        def format_docs(docs):
            context_string = ""
            for i, doc in enumerate(docs):
                context_string = context_string + "<Kontext Quelle=\"" + doc.metadata['source'] + ">\"" + doc.page_content + "</Kontext>"
                if i + 1 < len(docs):
                    context_string = context_string + "\n"
            return context_string

        # apply message templates to build user and system messages:
        file_templates = open(os.path.join(self.__own_directory,"message_templates.json"), 'r')
        self.__message_templates = json.load(file_templates)
        file_templates.close()
        if self.__text_generation_config.custom_system_message is None:
            system_message = self.__message_templates[message_template_name]["system"]
        else:
            system_message = self.__text_generation_config.custom_system_message
        if self.__text_generation_config.custom_user_message is None:
            user_message = self.__message_templates[message_template_name]["user"]
        else:
            user_message = self.__text_generation_config.custom_user_message

        # apply appropriate chat template to build prompt (documents and question are still missing):
        if self.__is_hf and self.__text_generation_config.custom_chat_template is not None:
            template = self.__text_generation_config.custom_chat_template.replace("{system-message}", system_message).replace("{user-message}", user_message)
            prompt = PromptTemplate(template=template, input_variables=["question", "context"])
        elif self.__is_hf:
            try:
                # print("trying chat template...")
                messages = [
                    {"role": "system_message",
                     "content": system_message},
                    {"role": "user_message",
                     "content": user_message},
                ]
                template = self.__tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt = PromptTemplate(template=template, input_variables=["question", "context"])
            except TemplateError:
                try:
                    # print("trying chat template without system role...")
                    messages = [
                        {"role": "user",
                         "content": system_message + "\n\n" + user_message},
                    ]
                    template = self.__tokenizer.apply_chat_template(messages, tokenize=False,
                                                                    add_generation_prompt=True)
                    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
                except TemplateError:
                    # print("trying without chat template...")
                    template = system_message + "\n\n" + user_message
                    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
        elif not self.__is_hf:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", user_message),
            ])
        else:
            raise Exception("No Models have been initialized yet. Use init_huggingface or init_openai first.")

        # construct chain, which takes question and documents and generates output:
        llm_chain = (
                RunnablePassthrough.assign(context=(lambda _: format_docs(context)))
                | prompt
                | llm
                | StrOutputParser()
        )

        # construct chain, which takes question, retrieves documents and passes both on to the llm_chain
        rag_chain = RunnableParallel(
            {'question': itemgetter("question")}
        ).assign(context=(lambda _: context)).assign(answer=llm_chain)

        return rag_chain

    def _get_random_docs(self, nr_of_docs):
        """
        Not intended to be used. Retrieves random documents from the database.

        :param nr_of_docs: Number of documents to retrieve
        :return: random documents from the database
        """
        nr_of_docs_total = self.__db.get_db_size()

        # Check if the total number of docs is large enough to get enough unique docs
        if nr_of_docs_total < nr_of_docs:
            raise ValueError("There are not enough documents in the database to get " + str(nr_of_docs) + " unique documents.")

        # Get the index of all docs to be retrieved
        docs_to_be_retrieved = random.sample(range(0, nr_of_docs_total), nr_of_docs)

        # Retrieve and return the documents
        docs = []
        for doc_i in docs_to_be_retrieved:
            doc = self.__db.get_one_doc(offset = doc_i)
            doc = Document(page_content=doc['documents'][0], metadata=doc['metadatas'][0])
            docs.append(doc)
        return docs

    def ask(self, question: str) -> dict:
        """
        Takes a question and returns the answer and all documents retrieved to answer the question.

        The returned dictionary contains the following keys:
            - `question` (str)
            - `answer` (str)
            - `context` (array of dict): The retrieved documents:
                - `page_content` (str): Content of the text fragment
                - `metadata` (dict):
                    - `source` (str): File path of the text fragment
                    - `page` (int): Page of the text fragment
                    - `chunk` (int): Position of the text fragment on the page
                    - `mod_date` (str): Modification date of the text fragments' file

        Note:
        The `init_huggingface` or `init_openai` method must be called once before this method can be called.

        :param question: Question to ask
        :return: A dictionary containing the question, answer and context
        """
        message_template_name = "German_2"
        context = self.__retriever_adapter(question)
        rag_chain = self.__prepare_rag_chain(message_template_name, context)
        return rag_chain.invoke({'question': str(question)})

    def _ask_with_custom_context(
            self,
            question: str,
            context: Union[list[dict], list[Document]],
    ) -> dict:
        """
        Not intended to be used. For debugging purposes only.
        """
        message_template_name = "German_2"
        rag_chain = self.__prepare_rag_chain(message_template_name, context)
        return rag_chain.invoke({'question': str(question)})


class _Embedding:
    """
    Adapter-class for Sentence Transformer, because methods of ChromaDB are different
    """

    def __init__(self, model_name, hf_transformers_cache_dir):
        self.embedding_model = SentenceTransformer(model_name, cache_folder=hf_transformers_cache_dir)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single string
        :param text: text to embed
        :return: embedding-vector
        """
        if not text.strip():
            print("Attempted to get embedding for empty text.")
            return []

        embedding_vector = self.embedding_model.encode(text)
        return embedding_vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings
        :param texts: list of texts to embed
        :return: list of embedding-vectors
        """
        embedding_vectors = []
        for text in texts:
            embedding_vectors.append(self.embed_query(text))
        return embedding_vectors


class _EmbeddingDummy:
    """
    Embedding Class, without actually producing embeddings
    """

    def embed_query(self, text: str) -> list[float]:
        return [0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embeddings.append([0])
        return embeddings


class _ReadFile:
    
    @classmethod
    def read(cls, file) -> list[list[Document]]:
        
        handlers = {
            'txt': cls.__read_txt,
            'pdf': cls.__read_pdf
            # Add more file extensions and handler methods here
        }

        file_extension = os.path.splitext(file['file_path'])[1][1:].lower()
        handler_function = handlers.get(file_extension)
        if handler_function:
            file['docs'] = handler_function(file['file_path'])
            file['mod_date_file'] = os.path.getmtime(file['file_path'])
        return file
    
    @staticmethod
    def __read_pdf(path: str) -> list[list[Document]]:

        chunk_size = 1000
        chunk_overlap = 200

        pdf = pdfium.PdfDocument(path)

        # read pdf page for page
        docs = []
        for i, page in enumerate(pdf):
            textpage = page.get_textpage()
            text = textpage.get_text_range().replace('￾', '')
            # add first 200 characters of next page to have overlap with next page
            if i < len(pdf) - 1:
                next_textpage = pdf[i + 1].get_textpage()
                text = text + "\r\n" + next_textpage.get_text_range().replace('￾', '')[:chunk_overlap]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            page_split = text_splitter.create_documents([text])
            docs.append(page_split)
            
        return docs
    
    @staticmethod
    def __read_txt(path: str) -> list[list[Document]]:

        chunk_size = 1000
        chunk_overlap = 200

        with open(path) as f:
            txt_content = f.read()
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        page_split = text_splitter.create_documents([txt_content])
        docs.append(page_split)

        return docs


class _DbAdapter:
    """
    Adapter Class for ChromaDB, which had significant performance losses after ~5.000.000 text fragments.

    Spreads texts across multiple DBs.
    """
    
    def __init__(
        self,
        persist_directory,
        embedding_function,
        n_retrievals
    ):
        self.__elements_per_db = 5000000
        self.__persist_directory = persist_directory
        self.__embedding_function = embedding_function
        self.__n_retrievals = n_retrievals
        self.__dbs = {}
        self.__db_sizes = {}
        # print(persist_directory)
        i = 0
        self.__init_db(i)
        while True:
            i = i+1
            if self.__does_db_exist(i):
                self.__init_db(i)
            else:
                break
    
    def retrieve(self, query: str):
        results = []
        for db in self.__dbs.values():
            results = results + db.similarity_search_with_relevance_scores(
                query = query, k = self.__n_retrievals)
        # sort list by similarity
        results = sorted(results, key=lambda x: x[1], reverse=True)
        # get the most similar documents
        results = results[:self.__n_retrievals]
        # get only the documents
        results = [x[0] for x in results]
        return results

    def get_db_size(self):
        size = 0
        for db_size in self.__db_sizes.values():
            size = size + db_size
        return size
    
    def get_id(self, id):
        for db in self.__dbs.values():
            if len(db.get(ids=[id])['ids']) != 0:
                return db.get(id)
        return {'ids':[], 'texts':[], 'metadatas':[]}
        
    def get_one_doc(self, offset = 0):
        current_position = 0
        for i_db, db in self.__dbs.items():
            db_size = self.__db_sizes[i_db]
            if current_position + db_size >= offset:
                return db.get(limit = 1, offset = offset - current_position)
            else:
                current_position = current_position + db_size
    
    def add_file(self, texts, metadatas, ids, embeddings = None):
        if len(ids) != 0:
            for i_db, db in self.__dbs.items():
                # if db didn't reach limit yet or file is already in db
                if self.__db_sizes[i_db] <= self.__elements_per_db or len(db.get(ids[0])['ids']) != 0:
                    if embeddings is not None:
                        db.add_texts(texts = texts, metadatas = metadatas, ids = ids, embeddings = embeddings)
                    else:
                        db.add_texts(texts = texts, metadatas = metadatas, ids = ids)
                    self.__db_sizes[i_db] = self.__db_sizes[i_db] + len(ids)
                    return
            n_new_db = len(self.__dbs)
            self.__init_db(n_new_db)
            if embeddings is not None:
                self.__dbs[n_new_db].add_texts(texts = texts, metadatas = metadatas, ids = ids, embeddings = embeddings)
            else:
                self.__dbs[n_new_db].add_texts(texts = texts, metadatas = metadatas, ids = ids)
            self.__db_sizes[n_new_db] = len(ids)

    def __does_db_exist(self, db_i: int):
        if os.path.exists(os.path.join(self.__persist_directory, str(db_i))):
            return True
        else:
            return False
    
    def __init_db(self, db_i: int):
        print("init db", db_i)
        db = Chroma(
            persist_directory = os.path.join(self.__persist_directory, str(db_i)),
            embedding_function = self.__embedding_function
        )
        self.__dbs[db_i] = db
        self.__db_sizes[db_i] = len(db.get(include=[])['ids'])
        
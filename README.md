**Prerequisits:**

- install CUDA

- install CUDNN

- AFTER installing the dependencies in requirements.txt:

On Windows:


`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

`pip install './to install before use/bitsandbytes-0.44.0.dev0-cp310-cp310-win_amd64.whl'`

On Linux:

`pip install torch torchvision torchaudio`

`pip install bitsandbytes`

- make sure, that TensorFlow is NOT installed:

`pip uninstall -y tensorboard tensorboard-data-server tensorboard-plugin-wit tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem`



------------------------




The package offers three classes to run retrieval augmented generation.


The **RetrievalConfig** class is primarily used to define the model which is used to
generate the vector representation of the texts, which will be retrieved. The name
of the model is passed to the class as the argument **embedding_model_name** when
initializing it. Besides that, there are also a number of other configuration options:

<details><summary>Click to expand</summary>

• The arguments **embedding_query_template** and **retrieval_query_template**
can be used to define a template that will be applied when generating the vector
representation of the retrievable text and the retrieving query. For example, the
embedding model E5 requires the prefixes “passage: ” and “query: ”. To en-
sure flexibility and applicability with different embedding models, these can be
freely configured. To fulfill the requirements of E5, embedding_query_template
would have to be `passage: {text}` and retrieval_query_template `query: {question}`.

• The arguments **db_directory** and **collection_name** are used to specify where
the text embeddings should be stored persistently.

• With **n_retrievals** it is possible to specify how many text fragments should be
retrieved and used as context to respond to the query, as different models have a
different context window size and therefore are able to process a different amount
of text fragments.
</details>



The **TextGenerationConfig** class is used to define the model which is used to generate
a response from the query and the retrieved texts. The name of the model is passed to
the class as the argument **text_generation_model_name** when initializing it. Other
possible configurations are:

<details><summary>Click to expand</summary>

• The argument **custom_chat_template** can define a custom template to format
the user and system message, for example with the correct BOT and EOT tokens
for the model. This might be necessary as different models use different chat templates.
And although usually the correct template is defined in the tokenizer_config.json
which is automatically downloaded with the model, sometimes it isn’t defined
there, or a wrong chat template was defined. In cases like that, an example of a
custom template is:

`system: {system-message} user: {user-message} assistant: `

“{system-message}” and “{user-message}” will be replaced by the actual system
and user message when the prompt for text generation is prepared by the program.

• The argument **custom_system_message** can be used to play around with the
model and provide an own system message, which will give the model instructions
on how to react to the provided input. As no further information will be inserted
in the system message, there is also no template to follow for the system message.

</details>


The most important class is the **RAG** class. While initializing it, the directory in which
the files for augmentation are situated is defined via the files directory argument.
The prototype supports .PDF and .TXT files.

After having initialized a RAG object, there are two options:

1. Choose **the init_openai** method to later perform the retrieval and generation
with models from OpenAI via their API. In this case, the open ai key
argument has to contain an API key for the OpenAI API.

2. Choose the **init_huggingface** method to later perform the retrieval and gener-
ation locally with models from the Hugging Face Hub. For some models
with access restrictions, the hf_hub_api_key is required to download the model,
to guarantee that access to the model has been granted. It is also possible to spec-
ify where the model parameters should be stored with **hf_transformers_cache_dir**.

In both cases, the RetrievalConfig and TextGenerationConfig objects are passed to the
method with the corresponding retrieval_config and text_generation_config argu-
ments.

If the init_huggingface function was chosen, it might take some time to prepare the
model parameters.
After that, the object is ready to work with the texts and the two functions update_files
and ask can be used:

• The function **update_files** reads all .PDF and .TXT files in the previously spec-
ified directory. It creates vector embeddings for the texts and stores them persis-
tently so that they can later be retrieved.

• The **ask** function performs the actual retrieval augmented generation. It takes
a question, retrieves the most relevant texts, and returns a generated answer to
the question.

An example can be found in run_rag.py

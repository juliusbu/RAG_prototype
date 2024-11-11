import os

class RetrievalConfig:
    def __init__(
            self,
            embedding_model_name: str = None,
            embedding_query_template: str = "{text}",
            retrieval_query_template: str = "{question}",
            db_directory: str = None,
            collection_name: str = "default_collection",
            n_retrievals: int = 3,
            **kwargs
    ):
        """
        :param embedding_model_name: Name of the model used to generate the vector representations.
        :param embedding_query_template: Optional. Template applied to document fragments before creating vector representation.
        :param retrieval_query_template: Optional. Template applied to question before creating vector representation.
        :param db_directory: Optional. Used to specify where the text embeddings should be stored persistently.
        :param collection_name: Optional. Used to specify where the text embeddings should be stored persistently.
        :param n_retrievals: Optional. Number of text fragments which are retrieved for every question.
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_query_template = embedding_query_template
        self.retrieval_query_template = retrieval_query_template
        if db_directory is None:
            self.db_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"db")
        else:
            self.db_directory = db_directory
        self.collection_name = collection_name
        self.n_retrievals = n_retrievals
        self.kwargs = kwargs


class TextGenerationConfig:
    def __init__(
            self,
            text_generation_model_name: str = None,
            custom_chat_template: str = None,
            custom_system_message: str = None,
            custom_user_message: str = None,
            **kwargs
    ):
        """

        :param text_generation_model_name: Name of the model used to generate the answer.
        :param custom_chat_template: Optional. Usually the model has a predefined chat template. Sometimes it isn't defined or wrong.
            Example format: "system: {system-message} user: {user-message} assistant:"
        :param custom_system_message: Optional. A good German-language system message is predefined.
        :param custom_user_message: Optional. A good German-language user message is predefined.
        """
        self.text_generation_model_name = text_generation_model_name
        self.custom_chat_template = custom_chat_template
        self.custom_system_message = custom_system_message
        self.custom_user_message = custom_user_message
        self.kwargs = kwargs

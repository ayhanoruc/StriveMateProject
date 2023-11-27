from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from typing import List,Tuple ,Union
import os 
import ast

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}


persist_directory = os.path.join(os.getcwd(), "artifact" ,"vector_db")
os.makedirs(persist_directory, exist_ok=True)

collection_name= "chat_record_summaries"

chat_summaries = [
    "User queried about best practices for coding in Python. AI offered guidelines on PEP 8, modularity, and comments.",
    "Inquiry on vegetarian recipes led to AI suggesting quinoa salad, lentil soup, and eggplant lasagna, with recipe details.",
    "Discussion on time management. AI recommended Pomodoro Technique, prioritization matrices, and digital tools like Trello.",
    "Conversation about space travel. AI explained the latest Mars mission, rover advancements, and SpaceX's role.",
    "Dialogue on learning guitar. AI provided beginner chords, maintenance tips, and online resource suggestions.",
    "Chat focused on mental health. AI highlighted importance of self-care routines, professional therapy, and mindfulness exercises.",
    "Exchange about fitness training. AI suggested HIIT workouts, balanced diets, and tracking progress with apps.",
    "Query about climate change effects. AI briefed on rising sea levels, extreme weather patterns, and global initiatives.",
    "User asked for book recommendations. AI suggested 'Sapiens,' '1984,' and 'The Alchemist,' with brief synopses.",
    "Conversation on AI ethics. AI summarized the significance of responsible AI use, privacy concerns, and bias prevention."
    "User inquires about weather, bot provides forecast and clothing advice.",
    "Bot assists user with resetting password, sends email confirmation.",
    "User asks for recipe suggestions, bot offers three based on dietary preferences.",
    "User struggles with app, bot troubleshoots and resolves issue.",
    "User requests travel tips, bot suggests destinations and safety tips.",
    "Bot guides user through meditation exercises for stress relief.",
    "User discusses book preferences, bot recommends latest bestsellers.",
    "Bot aids user in learning Spanish, provides practice exercises.",
    "User plans party, bot suggests themes, games, and music playlists.",
    "Bot explains quantum computing basics to curious user.",
    "User seeks investment advice, bot outlines risks and strategies.",
    "Bot and user engage in philosophical debate about artificial intelligence.",
    "User reports bug, bot logs the issue and informs dev team.",
    "Bot helps user with yoga poses, corrects posture through descriptions.",
    "User wants to learn guitar, bot sends tutorial videos.",
    "Bot assists in planning user's weekly meal prep, offers grocery list.",
    "User explores new podcasts, bot recommends based on interests.",
    "Bot educates user on climate change actions, provides sustainability tips.",
    "User inquires about historical events, bot gives a concise overview.",
    "Bot and user discuss the benefits of mindfulness and deep breathing.",
    "User said his name as Ayhan"
]
duplicated_list =[ 
    "User inquired about weather APIs; AI suggested OpenWeatherMap and provided implementation details.",
    "Conversation about Python's Flask vs Django; AI explained both frameworks' advantages.",
    "User sought advice on career change; AI discussed learning resources for software development.",
    "Discussion on healthy eating; AI provided recipes and nutritional information for balanced meals.",
    "User and AI exchanged views on renewable energy; AI emphasized solar and wind power benefits.",
    "Dialogue on stress management; AI recommended mindfulness techniques and regular exercise.",
    "User asked for book recommendations; AI listed top sci-fi novels and latest bestsellers.",
    "Conversation turned to space exploration; AI recounted latest Mars rover findings.",
    "AI assisted user with JavaScript code debugging; provided resources for further learning.",
    "User pondered learning a new language; AI compared difficulty levels and resources for various languages.",
    "User pondered learning a new language; AI compared difficulty levels and resources for various languages.",
    "User pondered learning a new language; AI compared difficulty levels and resources for various languages."
]

single_query = "learning a new language with methods at beginner difficulty levels"
multi_qry = ["learning a new language with methods at beginner difficulty levels","my spanish friend asked me about Python's Flask vs Django frameworks "]


class VectorRetriever:

    #I SHOULD IMPLEMENT DATA RELATED FUNCTIONS ALSO
    
    def __init__(self, model_name: str, model_kwargs: dict, encode_kwargs: dict, overwrite: bool = False) -> None:
        # we can pass different vector stores instead of built-in implemented-Chroma option. but this may require further consideration.
        self.vector_store = None
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.embedder = self.initialize_embedding_func()
        self.overwrite = overwrite
        

    def initialize_embedding_func(self):
        """
        Initializes the embedding function.

        :return: The initialized HuggingFaceEmbeddings object.
        """
        hf = HuggingFaceEmbeddings(
        model_name=self.model_name,
        model_kwargs=self.model_kwargs,
        encode_kwargs=self.encode_kwargs)
        embedding_dimension = hf.dict()['client'][1].get_config_dict()["word_embedding_dimension"]
        print("embedder initialized with dimension: ", embedding_dimension)

        return hf

    @staticmethod
    def drop_duplicates(raw_text_list):
        return list(set(raw_text_list))


    def initialize_vector_store(self, persist_directory, texts:List, collection_name:str,):
        """
        Initializes a Chroma vector store with the given texts and collection name and saves it to the persist_directory.

        Args:
            persist_directory (str): The directory to persist the vector store.
            texts (List[str]): The list of texts to be stored in the vector store.
            collection_name (str): The name of the collection.

        Returns:
            Chroma: The initialized Chroma vector store.
        """
        if self.overwrite:
            if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
                print("persist directory already exists")
            self.vector_store = Chroma.from_texts(texts=texts, 
                        embedding=self.embedder,
                        collection_name= collection_name,
                        persist_directory=persist_directory)
            self.vector_store.persist()
        
        else:
            self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embedder, collection_name=collection_name)
            

        print("vector store initialized successfully!")


    def similarity_search(self, query: Union[str, List[str]], k:int = 3 )->List[Tuple[str, float]]:
        """
        Performs a similarity search on the given query.

        Parameters:
        - query (Union[str, List[str]]): The query to search for. It can be either a single string or a list of strings.
        - k (int): The number of results to return. Default is 3.

        Returns:
        - List[Tuple[str, float]]: A list of tuples containing the similarity scores and their corresponding results.

        Raises:
        - ValueError: If the query is neither a string nor a list of strings.
        """

        if isinstance(query, list):
            query = "-".join(query) # concatenate list items to a string
        if isinstance(query, str):
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            print("relevant documents fetched successfully")
            return results
        else: 
            raise ValueError("query must be a string or a list of strings")

    
    def add_new_texts(self, texts: List[str]):
        """Adds new texts to the vector store.

        :param texts: A list of strings representing the texts to be added.
        :type texts: List[str]
        :return: None"""

        try: 
            texts = VectorRetriever.drop_duplicates(texts) 
            self.vector_store.add_texts(texts)
            self.vector_store.persist()
            print("new raw texts added to the vector store")

        except Exception as e:
            print("error occured when adding texts to vector store: ", e)





if __name__ == "__main__":


    PROJECT_ROOT = r"C:\Users\ayhan\Desktop\StriveMateProject" 
    summaries_path = os.path.join(PROJECT_ROOT, "summaries.txt")
    with open(summaries_path, "r") as f:
        summaries = f.read()
        actual_list = ast.literal_eval(summaries)
        print(len(actual_list))
    
    # NEED TO FIX vectordb load bug.
    vector_retriever = VectorRetriever(model_name = model_name, model_kwargs= model_kwargs, encode_kwargs=encode_kwargs, overwrite=False)
    vector_retriever.initialize_vector_store(persist_directory=persist_directory, texts=actual_list, collection_name=collection_name)
    qry = "do you remember that you had suggested me a dinner and a pasta recipe. what were those recipes?"
    response = vector_retriever.similarity_search(query=qry, k=3)
    print(response)
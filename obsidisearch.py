import os
import yaml
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import ObsidianLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings
from halo import Halo


def load_config() -> dict:
    """Loads the config.yaml file and returns it as a dictionary."""
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


@Halo(text='Loading documents from obsidian', spinner='dots1', color='magenta')
def load_notebooks() -> list:
    """Loads the documents from the obsidian vault and returns them as a list."""
    loader = ObsidianLoader(config["obisidian_directory"])
    documents = loader.load()
    return documents


@Halo(text='Creating vector database', spinner='dots1', color='magenta')
def create_vector_database(documents, embeddings) -> None:
    """Creates a vector database from the documents and persists it."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=texts, embedding_function=embeddings, persist_directory=config["persistence_directory"])
    vectordb.persist()
    vectordb = None
    return


@Halo(text='Loading vector database', spinner='dots1', color='magenta')
def load_vector_database() -> Chroma:
    """Loads the vector database from the persistence directory and returns it."""
    vectordb = Chroma(persist_directory=config["persistence_directory"], embedding_function=embeddings)
    return vectordb


@Halo(text='Creating prompt', spinner='dots1', color='magenta')
def create_prompt() -> dict:
    """Creates the prompt for the language model and returns it as a dictionary."""
    prompt = PromptTemplate(
        template=config["prompt"], input_variables=["context", "question"]
    )
    return {"prompt": prompt}


@Halo(text='Searching...', spinner='dots1', color='magenta')
def run_query(query):
    print(qa.run(query))
    return



config = load_config()
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(config["persistence_directory"]):
    print("Persisted vector database not found.")
    documents = load_notebooks()
    create_vector_database(documents, embeddings)
else:
    print(f"Persisted vector database found at: `{config['persistence_directory']}`")

vectordb = Chroma(persist_directory=config["persistence_directory"], embedding_function=embeddings)
chain_type_kwargs = create_prompt()

# Gives the option to use a chat model instead of a language model.
# non-chat models are generally cheaper to run but may not provide as good results.
# see https://openai.com/pricing for help choosing a model.
if config["use_chat_model"] is True:
    llm = ChatOpenAI(
            openai_api_key=config["openai_api_key"],
            model_name=config["openai_model"],
            temperature=config["temperature"],
            )
else:
    llm = OpenAI(
            openai_api_key=config["openai_api_key"],
            model_name=config["openai_model"],
            temperature=config["temperature"],
            )

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    chain_type_kwargs=chain_type_kwargs
    )

cont = True
query = input("Enter a query: ")
while cont:
    run_query(query)
    print("--------------------")
    query = input("\nEnter a new query: ")
    if not query:
        cont = False

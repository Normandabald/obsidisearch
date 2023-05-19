import sys
import os
import yaml
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import ObsidianLoader
from langchain.prompts import PromptTemplate
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from halo import Halo

spinner = Halo(text='Loading', spinner='dots', color='magenta')

def load_config():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

config = load_config()
spinner.start("Loading embedding model...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
spinner.stop()


# Create the vector database only if it doesn't already exist. (You will need to delete the directory to recreate it.)
if not os.path.exists(config["persistence_directory"]):
    print("Persisted vector database not found.")
    spinner.start("Creating a new one...")
    loader = ObsidianLoader(config["obisidian_directory"])
    documents = loader.load()
    spinner.stop()
    print(f"Loaded {len(documents)} notes from obsidian")

    # Split the documents into chunks of 300 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create a vector database from the documents
    spinner.start("Creating vector database")
    vectordb = Chroma.from_documents(documents=texts, embedding_function=embeddings, persist_directory=config["persistence_directory"])
    spinner.stop()
    # Persist the vector database to disk
    vectordb.persist()
    vectordb = None
else:
    print(f"Persisted vector database found at: `{config['persistence_directory']}`")

# Load the vector database from disk
vectordb = Chroma(persist_directory=config["persistence_directory"], embedding_function=embeddings)

# Create a custom prompt.
prompt = PromptTemplate(
    template=config["prompt"], input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": prompt}

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

cont = "y"
while cont == "y":
    query = input("Enter a query: ")
    spinner.start("Searching...")
    result = qa.run(query)
    spinner.succeed(result)
    print("--------------------")
    if input("Enter a new query: "):
        cont = "y"

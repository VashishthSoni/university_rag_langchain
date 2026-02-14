from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load API KEY
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

model = ChatOpenAI(model='gpt-4o-mini') 

loader = PyPDFLoader('GITUni.pdf')

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

document = loader.load()

splitted_document = splitter.split_documents(documents=document)

vector_store = Chroma.from_documents(
    documents=splitted_document,
    embedding=embedding,
    collection_name='GIT_Uni',
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={'k':2})
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    print("Context:\n", context)
    prompt = f"""
    Answer the question using only the context below.
    If answer is not found, say 'Not mentioned in document'.

    Context:
    {context}

    Question:
    {query}
    """

    response = model.invoke(prompt)
    print("AI:", response.content)
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext,StorageContext, Document,load_index_from_storage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.readers.database import DatabaseReader
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
os.environ["TOKENIZERS_PARALLELISM"] = "false"

UPLOAD_FOLDER = '/createdPdf'
PATH='http://192.168.13.189:5002/chat?collectionName='

#For fetching username and password for login
def checkLoginCredentials(username,conn):
    cur = conn.cursor()
    cur.execute('SELECT password FROM admin.user WHERE username = %s', (username,))
        # Fetch one record and return result
    account = cur.fetchone()
    if account:
        password_rs = account[0]
        return password_rs
    else:
        return ""

# For fetching all events   
def listEvents(conn):
    cur = conn.cursor()
    cur.execute("SELECT id,event_name,collection_name,folder_name,chatboturl,question_generted FROM admin.chatbot")
    data = cur.fetchall()
    return data

#For inserting values into postgresDb
def insertEvent(conn,eventName,collectionName,directory):
    path=PATH+collectionName
    cur = conn.cursor()
    cur.execute(
    "INSERT INTO admin.chatBot (event_name, collection_name,folder_name,chatboturl) VALUES (%s, %s, %s, %s)", (eventName, collectionName,directory,path))
    conn.commit()

#For deleting values from chatbot table
def deleteEvent(conn,id):
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM admin.chatBot WHERE id = %s", (id,)
    )
    conn.commit()

def deleteQARow(conn,id):
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM admin.generated_questions WHERE id = %s", (id,)
    )
    conn.commit()

def deleteCollection(collectionName):
    db = chromadb.PersistentClient(path="chroma_db")
    db.delete_collection(name=collectionName)

def insertChat(conn,query,collectionName,response):
    path=PATH+collectionName
    cur = conn.cursor()
    cur.execute(
    "INSERT INTO admin.chatHistory (query, response,collection_name) VALUES (%s, %s, %s)", (query, response,collectionName))
    conn.commit()

def savePdfLocally(directory,upload_files,app):
    if not os.path.isdir(directory):
        os.mkdir(directory)
        app.config['UPLOAD_FOLDER'] = directory
        for file in upload_files:
            if file:
                file.save(os.path.join(directory, file.filename))
        directory_path = os.path.dirname(upload_files[0].filename)


def createVectorDb(folder_path,collectionName):
    global chat_engine
    reader = SimpleDirectoryReader(input_dir=folder_path )
    docs = reader.load_data()
    llm = LlamaCPP(
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=os.path.join(os.getcwd(), "llama-2-7b-chat.Q2_K.gguf"),
        temperature=0.0,
        max_new_tokens=1024,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=4096,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={
            "n_gpu_layers": 56,
        },  # I need to play with this and see if it actually helps
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
       
    )


    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.chunk_size = 800
   # service_context = ServiceContext.from_defaults(
     #   chunk_size=800, llm=llm, embed_model="local"
    #)
    db = chromadb.PersistentClient(path="chroma_db")
    chroma_collection = db.get_or_create_collection(collectionName)

# construct vector store
    vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs,service_context=Settings, storage_context=storage_context)
    #index.storage_context.persist("chroma_db")
    #data_generator = DatasetGenerator.from_documents(docs)
    #eval_questions = data_generator.generate_questions_from_nodes()
    #print(eval_questions)


def editVectorDb(folder_path,collectionName):
    global chat_engine
    reader = SimpleDirectoryReader(input_dir=folder_path )
    docs = reader.load_data()
    llm = LlamaCPP(
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=os.path.join(os.getcwd(), "llama-2-7b-chat.Q2_K.gguf"),
        temperature=0.0,
        max_new_tokens=1024,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=4096,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={
            "n_gpu_layers": 56
        },  # I need to play with this and see if it actually helps
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
       
    )

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.chunk_size = 800
    #service_context = ServiceContext.from_defaults(
    #    chunk_size=800, llm=llm, embed_model="local"
   # )
    db = chromadb.PersistentClient(path="chroma_db")
    chroma_collection = db.get_or_create_collection(collectionName)

# construct vector store
    # vector_store = ChromaVectorStore(
    # chroma_collection=chroma_collection
    # )
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # index = VectorStoreIndex.from_documents(docs,service_context=service_context, storage_context=storage_context)
    index=chroma_collection.add(docs)
    index.storage_context.persist("chroma_db")


def initializeChatEngine(collectionName):
    llm = LlamaCPP(
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=os.path.join(os.getcwd(), "llama-2-7b-chat.Q2_K.gguf"),
        temperature=0.0,
        max_new_tokens=4096,
        
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=4096,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={
            "n_gpu_layers": 56
            #"n_threads" : 10,
            #"n_threads_batch": 10
        },  # I need to play with this and see if it actually helps
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
       
    )
   # service_context = ServiceContext.from_defaults(
    #    chunk_size=800, llm=llm, embed_model="local"
   # )
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.chunk_size = 800
    db = chromadb.PersistentClient(path="chroma_db")
    chroma_collection = db.get_or_create_collection(collectionName)
# construct vector store
    vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store,persist_dir="chroma_db")
    stored_index = load_index_from_storage(storage_context=storage_context, service_context=Settings)
    #vector_store = ChromaVectorStore(chroma_collection=collectionName)
    #storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #index =load_index_from_storage(storage_context=storage_context,service_context=service_context)
    chat_engine = stored_index.as_chat_engine(
        chat_mode="context", verbose=True, similarity_top_k=2
    )
    print(stored_index)
    return chat_engine


def editEvent(conn,id):
    cur = conn.cursor()
    cur.execute(
        "select * FROM admin.chatBot WHERE id = %s", (id,)
    )
    data = cur.fetchone()
    
    # Close the cursor, as you've finished the query
    cur.close()
    return data

def generatefaqs(conn,collectionName):
    cur = conn.cursor()
    if collectionName is not None:
        search_pattern = '%' + collectionName + '%'
        cur.execute("SELECT * FROM admin.chathistory WHERE collection_name LIKE %s", (search_pattern,))
        data = cur.fetchall()
    else:
        data = []
    
    # Close the cursor, as you've finished the query
    cur.close()
    return data

def fetchQns(conn,collectionName):
    cur = conn.cursor()
    if collectionName is not None:
        search_pattern = '%' + collectionName + '%'
        cur.execute("SELECT * FROM admin.generated_questions WHERE collection_name LIKE %s", (search_pattern,))
        data = cur.fetchall()
    else:
        data = []
    
    # Close the cursor, as you've finished the query
    cur.close()
    return data

def generateQuestion(folder_path):
    
    reader = SimpleDirectoryReader(input_dir=folder_path )
    docs = reader.load_data()
    llm = LlamaCPP(
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=os.path.join(os.getcwd(), "llama-2-7b-chat.Q2_K.gguf"),
        temperature=0.0,
        max_new_tokens=1024,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=8192,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={
            "n_gpu_layers": 56,
        },  # I need to play with this and see if it actually helps
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
       
    )


    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.chunk_size = 800
    index = VectorStoreIndex.from_documents(docs,service_context=Settings)
    data_generator = DatasetGenerator.from_documents(docs)
    eval_questions = data_generator.generate_questions_from_nodes()
    return eval_questions

def dropdown(conn):
    cur = conn.cursor()
    cur.execute("SELECT collection_name FROM admin.chatbot")
    data = cur.fetchall()
    return data

def updateFlag(conn,collection_name):
    cur = conn.cursor()
    cur.execute("update admin.chatbot set question_generted='true' WHERE collection_name LIKE %s", (collection_name,))
    conn.commit()
    cur.close()

def updateQAs(conn,row_id,question,answer):
    cur = conn.cursor()
    query = "UPDATE admin.generated_questions SET query = %s, response = %s WHERE id = %s", (question, answer, row_id)
    print (query)
    cur.execute("UPDATE admin.generated_questions SET query = %s, response = %s WHERE id = %s", (question, answer, row_id))
    conn.commit()
    cur.close()

def insertGeneratedQuestions(question,response,conn,collection_name):
    cur = conn.cursor()
        # Insert each question into the database table   
    cur.execute("INSERT INTO admin.generated_questions (query,response,collection_name) VALUES (%s, %s, %s)", (question,response,collection_name))
        
    # Commit the transaction and close the database connection
    conn.commit()
    cur.close()

def trainVectorDb(reader,collectionName):
   
    
    search_pattern = '%' + collectionName + '%'
    query = "SELECT query,response FROM admin.generated_questions WHERE collection_name LIKE 'krishna'"
    documents = reader.load_data(query=query)
    llm = LlamaCPP(
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=os.path.join(os.getcwd(), "llama-2-7b-chat.Q2_K.gguf"),
        temperature=0.0,
        max_new_tokens=1024,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=4096,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={
            "n_gpu_layers": 56
        },  # I need to play with this and see if it actually helps
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
       
    )

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.chunk_size = 800
    #service_context = ServiceContext.from_defaults(
    #    chunk_size=800, llm=llm, embed_model="local"
   # )
    db = chromadb.PersistentClient(path="chroma_db")
    chroma_collection = db.get_or_create_collection(collectionName)
    print("count before", chroma_collection.count())
    #print (chroma_collection.get(0).)
    print([x.doc_id for x in documents])
   # chroma_collection.add(
  #  documents=documents,
   # ids=[x.doc_id for x in documents]
#)

    #deleteCollection(collectionName)
    #print("count laater", chroma_collection.count())
# construct vector store
    #chroma_collection = db.get_or_create_collection(collectionName)
   # print("count after deletion", chroma_collection.count())
    vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents,service_context=Settings, storage_context=storage_context)
    index.storage_context.persist("chroma_db")
    print("count after", chroma_collection.count())
    
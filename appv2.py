from flask import Flask
from flask import Flask, render_template, redirect, url_for, request,session,flash
import psycopg2
from llama_index import VectorStoreIndex, ServiceContext,StorageContext, Document,load_index_from_storage
from llama_index.llms import LlamaCPP
from llama_index import SimpleDirectoryReader
import os
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import chromadb
from llama_index.vector_stores import ChromaVectorStore

app = Flask(__name__)
app.secret_key = 'keltron_chatbot'
UPLOAD_FOLDER = '/data/hfllama/createdPdf'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DB_HOST = "localhost"
DB_NAME = "chat_bot"
DB_USER = "postgres"
DB_PASS = "password"
 
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

@app.route("/")
def home():
    return "Hello, Keltron!"

def createVectorDb(folder_path,collectionName):
    global chat_engine
    reader = SimpleDirectoryReader(input_dir=folder_path )
    docs = reader.load_data()
    llm = LlamaCPP(
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=os.path.join(os.getcwd(), "/data/hfllama/llama-2-7b-chat.Q2_K.gguf"),
        temperature=0.0,
        max_new_tokens=1024,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=4096,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={
            "n_gpu_layers": 23,
        },  # I need to play with this and see if it actually helps
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
       
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=800, llm=llm, embed_model="local"
    )
    db = chromadb.PersistentClient(path="/data/hfllama/chroma_db")
    chroma_collection = db.get_or_create_collection(collectionName)

# construct vector store
    vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs,service_context=service_context, storage_context=storage_context)
    index.storage_context.persist("/data/hfllama/chroma_db")
   


def initializeChatEngine(collectionName):
    llm = LlamaCPP(
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=os.path.join(os.getcwd(), "/data/hfllama/llama-2-7b-chat.Q2_K.gguf"),
        temperature=0.0,
        max_new_tokens=1024,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=4096,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={
            "n_gpu_layers": 23,
        },  # I need to play with this and see if it actually helps
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
       
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=800, llm=llm, embed_model="local"
    )
    
    db = chromadb.PersistentClient(path="/data/hfllama/chroma_db")
    chroma_collection = db.get_or_create_collection(collectionName)
# construct vector store
    vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store,persist_dir="/data/hfllama/chroma_db")
    stored_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
    #vector_store = ChromaVectorStore(chroma_collection=collectionName)
    #storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #index =load_index_from_storage(storage_context=storage_context,service_context=service_context)
    chat_engine = stored_index.as_chat_engine(
        chat_mode="context", verbose=True, similarity_top_k=2
    )
    print(stored_index)
    return chat_engine

@app.route("/event",methods=['GET', 'POST'])
def event():
   if request.method == 'POST' :
      eventName = request.form['eventName']
      collectionName = request.form['collectionName']
      print(eventName)
      print(collectionName)
      upload_files = request.files.getlist('files')
      if not upload_files:
        return 'No selected file'
      else:
        directory=UPLOAD_FOLDER+ "/"+collectionName
    # Get the directory path from the file's path
        if not os.path.isdir(directory):
         os.mkdir(directory)
        app.config['UPLOAD_FOLDER'] = directory
        for file in upload_files:
            if file:
                file.save(os.path.join(directory, file.filename))
        directory_path = os.path.dirname(upload_files[0].filename)
        print(directory_path)
        cur = conn.cursor()
        cur.execute(
        "INSERT INTO admin.chatBot (event_name, collection_name,folder_name) VALUES (%s, %s, %s)", (eventName, collectionName,directory))
        conn.commit()
        createVectorDb(directory,collectionName)
        return redirect(url_for('chat'))
   return render_template('event.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    cur = conn.cursor()
     # Check if account exists using MySQL
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        print(password)
        cur.execute('SELECT password FROM admin.user WHERE username = %s', (username,))
        # Fetch one record and return result
        account = cur.fetchone()
        if account:
          password_rs = account[0]
          print("Password from db is" + password_rs)
            # If account exists in users table in out database
          if password_rs == password:
                # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['username'] = username
                # Redirect to home page
            #return redirect(url_for('list'))
            return redirect(url_for('list'))
          else:
                # Account doesnt exist or username/password incorrect
            flash('Incorrect username/password')
        else:
            # Account doesnt exist or username/password incorrect
            flash('Incorrect username/password')
    return render_template('login.html')

@app.route('/list')
def list():
    cur = conn.cursor()
    cur.execute("SELECT event_name,collection_name,folder_name FROM admin.chatbot")
    data = cur.fetchall()
    return render_template('list.html', data=data)

@app.route("/get", methods=["GET","POST"])
def chat_response():
    message = request.form["message"]
    global chat_engine
    response = chat_engine.chat(message)
    return response.response


@app.route('/chat')
def chat():
    global chat_engine
    chat_engine=initializeChatEngine('krishna')
    print ("chat engine initialized")
    return render_template('chat.html')
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)
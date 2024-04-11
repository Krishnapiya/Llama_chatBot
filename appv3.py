from flask import Flask
from flask import Flask, render_template, redirect, url_for, request,session,flash
import psycopg2
import os

from chatBotCreator import *;

app = Flask(__name__)
app.secret_key = 'keltron_chatbot'
UPLOAD_FOLDER = '/data/hfllama/createdPdf'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DB_HOST = "localhost"
DB_NAME = "chat_bot"
DB_USER = "postgres"
DB_PASS = "password"


#create connection
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
chat_engines = {}

@app.route("/")
def home():
    return "Hello, Keltron!"


@app.route("/event",methods=['GET', 'POST'])
def event():
   if request.method == 'POST' :
      eventName = request.form['eventName']
      collectionName = request.form['collectionName']
      upload_files = request.files.getlist('files')
      if not upload_files:
        return 'No selected file'
      else:
        directory=UPLOAD_FOLDER+ "/"+collectionName
        savePdfLocally(directory,upload_files,app)
        insertEvent(conn,eventName,collectionName,directory)
        createVectorDb(directory,collectionName)
        return redirect(url_for('list'))
   return render_template('event.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

     # Check if account exists using POSTGRES
        password_rs=checkLoginCredentials(username,conn)  
        # If account exists in users table in out database
        if password_rs == password and password_rs!="":
                # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['username'] = username
                # Redirect to home page
            #return redirect(url_for('list'))
            return redirect(url_for('list'))
        else:
                # Account doesnt exist or username/password incorrect
            flash('Incorrect username/password')
    return render_template('login.html')

@app.route('/list')
def list():
    data=listEvents(conn)
    return render_template('list.html', data=data)

@app.route("/get", methods=["GET","POST"])
def chat_response():
    message = request.form["message"]
    collection_name = request.form["collectionName"]
    print (collection_name)
    if collection_name not in chat_engines:
        return "Error: Chat engine not initialized for collection name {}".format(collection_name)

    response = chat_engines[collection_name].chat(message)
    #response = chat_engine.chat(message)
    return response.response
#fggfg

@app.route('/chat')
def chat():
    collectionName = request.args.get('collectionName')
    if collectionName not in chat_engines:
        chat_engines[collectionName] = initializeChatEngine(collectionName)
        print("Chat engine initialized for collection name:", collectionName)
    return render_template('chat.html',data=collectionName)
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, render_template, redirect, url_for, request,session,flash,current_app, g,jsonify,make_response
import psycopg2
import os
import threading 
import pdfkit
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

from chatBotCreator import *;

app = Flask(__name__)
app.secret_key = 'keltron_chatbot'
UPLOAD_FOLDER = 'createdPdf'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#DB_HOST = "localhost"
DB_HOST="192.168.13.189"
DB_PORT =  5432
DB_NAME = "chat_bot"
DB_USER = "postgres"
DB_PASS = "password"
lock = Lock()
#tets hhkjhjhh
conn = psycopg2.connect(dbname=DB_NAME,port=DB_PORT, user=DB_USER, password=DB_PASS, host=DB_HOST)

chat_engines = {}
# Use a lock for the chat_engines dictionary
chat_engines_lock = threading.Lock()
@app.route("/")
def home():
    return render_template('welcome.html')


@app.route("/event",methods=['GET', 'POST'])
def event():
   if request.method == 'POST' :
      eventName = request.form['eventName']
      collectionName = request.form['collectionName']
      upload_files = request.files.getlist('files')
      if not upload_files:
        return 'No selected file'
      else:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
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
    with app.app_context():
        data=listEvents(conn)
    return render_template('list.html', data=data)

@app.route("/get", methods=["GET","POST"])
def chat_response():
    message = request.form["message"]
    collection_name = request.form["collectionName"]
    print (collection_name)
    with chat_engines_lock:
        if collection_name not in chat_engines:
            return "Error: Chat engine not initialized for collection name {}".format(collection_name)
    with lock:
        response =  chat_engines[collection_name].chat(message)
        insertChat(conn,message,collection_name,response.response)
    pass
    #response = chat_engine.chat(message)
    return response.response

@app.route('/deleteCollection', methods=['POST'])
def delete_row():
    row_id = request.form.get('row_id')
    collection_name=request.form.get('collection_name')
    # Perform any necessary actions with the row_id (e.g., delete from database)
    deleteEvent(conn,row_id)
    deleteCollection(collection_name)
    return jsonify({'status': 'success'})

@app.route('/deleteQA', methods=['POST'])
def delete_qa():
    row_id = request.form.get('row_id')
    print(row_id)
    deleteQARow(conn,row_id)
    return jsonify({'status': 'success'})


@app.route('/chat')
def chat():
    
    collectionName = request.args.get('collectionName')
    if collectionName not in chat_engines:
        chat_engines[collectionName] = initializeChatEngine(collectionName)
        print("Chat engine initialized for collection name:", collectionName)
    return render_template('chat.html',data=collectionName)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/generate',methods=['GET', 'POST'])
def generate_question():
    folder_path=request.args.get('folder_path')
    collection_name=request.args.get('collection_name')
    print(folder_path)
    # Perform any necessary actions with the row_id (e.g., delete from database)
    questions = generateQuestion(folder_path)
    updateFlag(conn,collection_name)
    chat_engines[collection_name] = initializeChatEngine(collection_name)
    for question in questions:
        if '?' in question:
            with lock:
                response =  chat_engines[collection_name].chat(question)
                insertGeneratedQuestions(question,response.response,conn,collection_name)
    return render_template('questions.html',data=questions)

@app.route('/faq',methods=['GET', 'POST'])
def faq():
    selected_option = request.form.get('selected_option')
    faqs = [] 
    if selected_option is not None :
         faqs = generatefaqs(conn, selected_option)
         print(faqs)
        # Set a default option or handle the case appropriately
    # Generate FAQ data based on the selected option
    
    # Define dropdown options
    dropdown_options = dropdown(conn)
    collection_names = [row[0] for row in dropdown_options]
    
    # Pass FAQ data and dropdown options to the template
    return render_template('faq.html', data=faqs, dropdown_options=collection_names,selected_option=selected_option)

@app.route('/generatedQuestions',methods=['GET', 'POST'])
def generatedQuestions():
    selected_option = request.form.get('selected_option')
    qns = [] 
    print (selected_option)
    if selected_option is not None :
         qns = fetchQns(conn, selected_option)
         print(qns)
        # Set a default option or handle the case appropriately
    # Generate FAQ data based on the selected option
    
    # Define dropdown options
    dropdown_options = dropdown(conn)
    collection_names = [row[0] for row in dropdown_options]
    
    # Pass FAQ data and dropdown options to the template
    #return render_template('generated.html', data=qns, dropdown_options=collection_names,selected_option=selected_option)
    return render_template('generated.html', data=qns, dropdown_options=collection_names,selected_option=selected_option)
    
    

@app.route('/updateQA',methods=['GET', 'POST'])
def updateQA():
    row_id= request.form.get('row_id')
    question =request.form.get('question')
    answer= request.form.get('answer')
    print(answer)
    updateQAs(conn,row_id,question,answer)
    return jsonify({'status': 'success'})


@app.route('/trainDb',methods=['GET', 'POST'])
def trainDb():
    selected_option = request.args.get('selected_option')
    print(selected_option)
    reader = DatabaseReader(
    uri="postgresql://postgres:password@192.168.13.189:5432/chat_bot")
    trainVectorDb(reader,selected_option)
    dropdown_options = dropdown(conn)
    collection_names = [row[0] for row in dropdown_options]
    
    # Pass FAQ data and dropdown options to the template
    #return render_template('generated.html', data=qns, dropdown_options=collection_names,selected_option=selected_option)
    return render_template('generated.html', dropdown_options=collection_names,selected_option=selected_option)


@app.route('/saveQA',methods=['GET', 'POST'])
def saveQA():
    selected_option= request.form.get('selectedOption')
    question =request.form.get('question')
    answer= request.form.get('answer')
    print(answer)
    insertGeneratedQuestions(question,answer,conn,selected_option)
    return jsonify({'status': 'success'})
    
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5002, debug=True)
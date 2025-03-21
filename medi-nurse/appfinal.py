import os
import pickle
import re
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv('.env')
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize Flask app
app = Flask(__name__)

# Load depression detection models
model_rf_q1 = pickle.load(open('model.pkl', 'rb'))  
model_rf_q3 = pickle.load(open('modelq3.pkl', 'rb'))  
model_rf_q4 = pickle.load(open('modelq4.pkl', 'rb')) 
model_rf_q5 = pickle.load(open('modelq5.pkl', 'rb')) 
model_rf_q6 = pickle.load(open('modelq6.pkl', 'rb'))  

cv_q1 = pickle.load(open('cv-transform.pkl', 'rb'))  
cv_q3 = pickle.load(open('cv-transformq3.pkl', 'rb'))  
cv_q4 = pickle.load(open('cv-transform4.pkl', 'rb')) 
cv_q5 = pickle.load(open('cv-transform5.pkl', 'rb'))  
cv_q6 = pickle.load(open('cv-transform6.pkl', 'rb')) 

# Stopwords and stemmer
all_stopwords = stopwords.words('english')
stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(response, cv, model):
    nresponse = re.sub('[^a-zA-Z]', ' ', response)
    nresponse = nresponse.lower().split()
    nresponse = [stemmer.stem(word) for word in nresponse if word not in set(all_stopwords)]
    nresponse = ' '.join(nresponse)
    new_corpus = [nresponse]
    new_X_test = cv.transform(new_corpus).toarray()
    prediction = model.predict(new_X_test)
    return prediction

# Load LLM for RAG-based chatbot
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain for chatbot
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Routes
@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/test')
def test():
    return render_template('test.html', active_page='take-test')

@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract responses for all questions
    feeling = request.form['feeling']
    life = request.form['life']
    energy = request.form['energy']
    sleep = request.form['sleep']
    thoughts = request.form['thoughts']
    physical_changes = request.form['physical_changes']
    
    # Combine first two inputs for the first model
    combined_input_q1 = feeling + " " + life
    
    # Predictions
    prediction_q1 = preprocess_text(combined_input_q1, cv_q1, model_rf_q1)
    prediction_q3 = preprocess_text(energy, cv_q3, model_rf_q3)
    prediction_q4 = preprocess_text(sleep, cv_q4, model_rf_q4)
    prediction_q5 = preprocess_text(thoughts, cv_q5, model_rf_q5)
    prediction_q6 = preprocess_text(physical_changes, cv_q6, model_rf_q6)
    
    # Combine predictions from all models
    predictions = [prediction_q1[0], prediction_q3[0], prediction_q4[0], prediction_q5[0], prediction_q6[0]]
    depressed_count = sum(predictions)
    
    # Decision logic
    if depressed_count >= 4:
        result = "Depressed"
    elif 2 <= depressed_count < 4:
        result = "Moderately Depressed"
    else:
        result = "Not Depressed"
    
    return render_template('combined.html', result=result, active_page='take-test')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get("query")
    response = qa_chain.invoke({'query': user_query})
    return jsonify({
        "result": response["result"],
        "source_documents": [doc.metadata for doc in response["source_documents"]]
    })

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

#loading our models
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

# stopwords and stemmer
all_stopwords = stopwords.words('english')
stemmer = PorterStemmer()

# new corpus
def preprocess_text(response, cv, model):
    nresponse = re.sub('[^a-zA-Z]', ' ', response)
    nresponse = nresponse.lower().split()
    nresponse = [stemmer.stem(word) for word in nresponse if not word in set(all_stopwords)]
    nresponse = ' '.join(nresponse)
    
   
    new_corpus = [nresponse]
    new_X_test = cv.transform(new_corpus).toarray()
    
 #prediction
    prediction = model.predict(new_X_test)
    return prediction

# home page route
@app.route('/')
def index():
    return render_template('comb.html')

# submission and display route
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
    
    # predictions 
    prediction_q1 = preprocess_text(combined_input_q1, cv_q1, model_rf_q1)
    prediction_q3 = preprocess_text(energy, cv_q3, model_rf_q3)
    prediction_q4 = preprocess_text(sleep, cv_q4, model_rf_q4)
    prediction_q5 = preprocess_text(thoughts, cv_q5, model_rf_q5)
    prediction_q6 = preprocess_text(physical_changes, cv_q6, model_rf_q6)

    # Combine predictions from all models
    predictions = [prediction_q1[0], prediction_q3[0], prediction_q4[0], prediction_q5[0], prediction_q6[0]]
    
    # Final decision logic based on multiple model outputs
    depressed_count = sum(predictions)  # Count how many predictions indicate depression

    if depressed_count >= 4:  # Majority indicates "Depressed"
        result = "Depressed"
    elif 2 <= depressed_count < 4:  # Moderate signs of depression
        result = "Moderately Depressed"
    else:
        result = "Not Depressed"
    
    # Render the template with the final prediction result
    return render_template('comb.html', result=result, active_page='take-test')

if __name__ == '__main__':
    app.run(debug=True)
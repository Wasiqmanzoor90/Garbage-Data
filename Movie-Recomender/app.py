from flask import Flask, request, render_template
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)


# Load saved model objects

with open('imdb_genre_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']                # Trained RandomForestClassifier
ct = data['column_transformer']      # Preprocessing transformer (e.g., OneHotEncoder)
le_genre = data['label_encoder']     # LabelEncoder for genres


# Define Flask route

@app.route('/', methods=['GET', 'POST'])
def home():
    msg = ""  # Message to display in the webpage

    if request.method == 'POST':
        # Get user input from form
        title = request.form['Series_Title']
        year = request.form['Released_Year']

        # Validate year input
        if not year.isdigit():
            return render_template('index.html', msg="Invalid year. Please enter a number.")

   
        # Prepare input as a DataFrame
    
        df = pd.DataFrame([{
            'Series_Title': title,
            'Released_Year': int(year)
        }])

     
        # Transform features and predict
      
        X = ct.transform(df)              # Apply preprocessing (e.g., one-hot encoding)
        pred = model.predict(X)           # Make prediction
        genre = le_genre.inverse_transform(pred)[0]  # Convert numeric label back to genre

        msg = f"Predicted Genre: {genre}"

    # Render HTML template with message
    return render_template('index.html', msg=msg)


# Run the Flask app

app.run(debug=True)

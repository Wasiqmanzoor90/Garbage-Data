from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
#flask app
@app.route('/', methods=['POST', 'GET'])
def Home():
    if request.method == 'POST':
        with open('pickle_logistic.pkl', 'rb') as file:
            model = pickle.load(file)
        
        with open('label_encoder.pkl', 'rb') as file:
            le = pickle.load(file)

        #gets all values the user submitted in the form (they come as strings)
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        predict = model.predict(final_features)
        result = le.inverse_transform(predict)[0]

        return render_template('index.html', msg=result)

    # GET request
    return render_template('index.html')



app.run(debug=True)

# ml to pickle
# import pickle

# with open('pickle_knn.pkl','wb') as file:
#     pickle.dump(knn,file)
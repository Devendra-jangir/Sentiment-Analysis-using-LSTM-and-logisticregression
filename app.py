from flask import Flask, render_template, request
import model  # Import the model functions

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = model.predict_sentiment(message)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

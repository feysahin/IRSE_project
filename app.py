from flask import Flask, render_template, request
from data import Models

app = Flask(__name__)

Models = Models()

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    return render_template('data.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':

        d_size = request.form.get('d_size')
        print('\nData-set size: %' + d_size)

        tr_size = request.form.get('tr_size')
        print('Train-set size: ' + tr_size)

        n_epoch = request.form.get('n_epoch')
        print('# epoch: ' + n_epoch)

        models = request.form.getlist('model_names')
        print('Models: ', models, '\n')

    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

    
if __name__ == '__main__':
    app.run(debug = True)


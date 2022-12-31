from flask import Flask, render_template, request
from data import Models
from machine_learning import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



app = Flask(__name__)

def plot_png2():
   fig = Figure()
   axis = fig.add_subplot(1, 1, 1)
   xs = np.random.rand(100)
   ys = np.random.rand(100)
   axis.plot(xs, ys)
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

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

        predictions, y_test = train_model(models[0], int(d_size), int(tr_size))
        acuuracy_v, precision_v, recall_v, f1_v = accuracy(predictions, y_test)
        print('Accuracy: ', acuuracy_v)
        print('Precision: ', precision_v)

    return render_template('index.html', acuuracy_v=acuuracy_v, precision_v=precision_v, recall_v=recall_v, f1_v=f1_v)

@app.route('/')
def index():
    return render_template('index.html')

    
if __name__ == '__main__':
    app.run(debug = True)


from flask import Flask, render_template, request, send_file, make_response, url_for, Response
from data import Models
from machine_learning import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import io

matplotlib.use('Agg')

app = Flask(__name__)

Models = Models()

@app.route('/contact')
def contact():
    return render_template('contact.html')

 

"""
@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if request.method == 'POST':
        d_size = request.form.get('d_size')
        print('\nData-set size: %' + d_size)
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')

    # Save the plot to a temporary buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return render_template('data.html', plot_url=buf.getvalue())

"""

#Matplotlib page
@app.route('/visualize', methods=("POST", "GET"))
def visualize():
    return render_template('data.html',
                           PageTitle = "Visualize")



@app.route('/piechart.png/<data_size_value>', methods=['GET'])
def plot_pie_chart(data_size_value):
    print(data_size_value)
    fig = pie_chart(int(data_size_value))
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')




@app.route('/boxplot.png', methods=['GET'])
def plot_box_plot():
    data_size = request.args.get('data_size')
    column_name = request.args.get('column')

    fig = boxplot_data(int(data_size),column_name )
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')



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

    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

    
if __name__ == '__main__':
    app.run(debug = True)


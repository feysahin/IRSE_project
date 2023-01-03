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

 

#Matplotlib page
@app.route('/visualize', methods=("POST", "GET"))
def visualize():
    return render_template('data.html',
                           PageTitle = "Visualize")



@app.route('/piechart.png', methods=['GET'])
def plot_pie_chart():
    data_size_value = request.args.get('data_size')
    preprocess_data = request.args.get('preprocess')
    fig = pie_chart(int(data_size_value), preprocess_data)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')




@app.route('/boxplot.png', methods=['GET'])
def plot_box_plot():
    data_size = request.args.get('data_size')
    column_name = request.args.get('column')
    preprocess_data = request.args.get('preprocess')

    fig = boxplot_data(int(data_size),column_name,preprocess_data)
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

        model, x, x_test, y, y_test = train_model(models[0], int(d_size), int(tr_size))
        test_accuracy, test_precision, test_recall, test_f1 = compute_accuracy(model, x_test, y_test)
        train_accuracy, train_precision, train_recall, train_f1 = compute_accuracy(model, x, y)

        print('Test Accuracy: ', test_accuracy)
        print('Test Precision: ', test_precision)
        print('Test Recall: ', test_recall)
        print('Test F1: ', test_f1)

        print('Train Accuracy: ', train_accuracy)
        print('Train Precision: ', train_precision)
        print('Train Recall: ', train_recall)
        print('Train F1: ', train_f1)

    return render_template('train_results.html', model_name=models[0] , test_accuracy=test_accuracy, test_precision=test_precision,
                           test_recall=test_recall, test_f1=test_f1, train_accuracy=train_accuracy,
                           train_precision=train_precision, train_recall=train_recall, train_f1=train_f1)
@app.route('/')
def index():
    return render_template('index.html')

    
if __name__ == '__main__':
    app.run(debug = True)


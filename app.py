import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file, Response, url_for
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit

import io
import base64
from threading import Thread

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor

from jax.tree_util import tree_map
from phonax.data_utils import to_f32
from phonax.trained_models import NequIP_JAXMD_uniIAP_model
from phonax.phonons import (
    atoms_to_ext_graph,
    plot_phonon_DOS,
    dynamical_matrix
)

from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.backends.backend_agg import FigureCanvasAgg

from plotting import plot_bands, mpl_to_html, render_mpl
from tests.test_model import test_hessian
from phonax_web import predict_hessian_matrix
from events import event_emitter


app = Flask(__name__)
socketio = SocketIO(app)
event_emitter.init_socketio(socketio)


UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.after_request
def close_mpl_plot(response):
    """This prevents memory leakage; Matplotlib's pyplot API is stateful, which
    can be a burden for a website that runs for a while.
    """
    import matplotlib.pyplot as plt
    plt.close('all')
    return response

@app.route('/task')
def task():
    return render_template('task.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Source:
    # https://matplotlib.org/gallery/lines_bars_and_markers/simple_plot.html
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()  
    ax.plot(t, s)  
    ax.set(
        xlabel='time (s)',
        ylabel='voltage (mV)',  
        title='About as simple as it gets, folks'
    )

    # Add file upload form to the HTML
    upload_form = '''
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload and Process">
        </form>
    '''

    # Handle file upload and processing
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            parser = CifParser(file_path)
            structure = parser.parse_structures()[0]

            # Print some basic information about the loaded structure
            print(f"Loaded structure: {structure.composition.reduced_formula}")
            print(f"Number of sites: {len(structure)}")
            print(f"Lattice parameters: {structure.lattice.abc}")
            print(f"Space group: {structure.get_space_group_info()}")

            model_fn, params, num_message_passing, r_max = NequIP_JAXMD_uniIAP_model(os.path.join(os.getcwd(), 'trained-models'))

            asecell = AseAtomsAdaptor.get_atoms(structure)

            # Get the coordinates of the atoms within asecell
            atom_positions = asecell.get_positions()
            print("Atom positions:\n", atom_positions)

            print ('building graph')
            graph = atoms_to_ext_graph(asecell, r_max, num_message_passing)   
            graph = tree_map(to_f32, graph)
            atoms = asecell

            print ('predicting hessian')

            # H = test_hessian(graph)
            H = predict_hessian_matrix(params,model_fn,graph)

            fig = plot_bands(atoms, graph, H, npoints=1000)
            fig_html = mpl_to_html(fig)

            return render_template('index.html', fig=fig_html)
        
    # If it's a GET request, just show the upload form
    return render_template('index.html')

@app.route("/phonon-plot.svg")
def plot_svg():
    """ renders the plot on the fly.
    """

    fig = process_file('SiO2.cif')

    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")

@app.route("/phonon.png")
def plot_png(num_x_points=50):
    """ renders the plot on the fly.
    """
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [np.random.randint(1, 30) for x in x_points])

    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")

@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.init_app(app)
    socketio.run(app, debug=True)
    app.config['socketio'] = socketio
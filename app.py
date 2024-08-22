import os
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file
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
    predict_hessian_matrix,
    atoms_to_ext_graph,
    plot_bands,
    plot_phonon_DOS,
    dynamical_matrix
)

app = Flask(__name__)
socketio = SocketIO(app)

UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def process_file(file_path):
    # Load the structure from a CIF file
    # Replace 'path_to_your_cif_file.cif' with the actual path to your CIF file
    parser = CifParser(file_path)
    structure = parser.get_structures()[0]
            # socketio.emit('progress', {'progress': progress})

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


    graph = atoms_to_ext_graph(asecell, r_max, num_message_passing)   
    graph = tree_map(to_f32, graph)
    atoms = asecell

    H = predict_hessian_matrix(params,model_fn,graph, socket=True)

    fig = plot_bands(atoms, graph, H, npoints=1000)

    return send_file(fig, mimetype='image/png')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            # filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Start processing in a background thread
            Thread(target=process_and_emit, args=(file_path,)).start()
            
            return render_template('processing.html')

    return render_template('upload.html')


def process_and_emit(file_path):
    plot_data = process_file(file_path)
    
    # Delete the file
    os.remove(file_path)
    
    # Emit completion event with plot data
    socketio.emit('complete', {'plot_data': plot_data})

@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.run(app, debug=True)
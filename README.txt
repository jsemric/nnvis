NNvis

Visualisation of neural networks.


1. Requirements

Python >= 3.5


2. Install

$ pip install -r requirements.txt`


3. Usage:

Running the model and collecting its data.
The -o option specifies where the NN data will be stored, in this case nndump.json

$ python cifar_model.py -o nndump.json

Print the json in the human readable format.

$ python print_json.py nndump.json


NNvis
=====

Visualization of neural networks.

Requirements
============

 - **Python** - version >= 3.5
 - **R** - version >= 3.4.4
 - **MongoDB** (optional)

Install
=======

`$ pip install -r requirements.txt`

Usage
=====

Running the model and collecting the data during training.
The `-o` option specifies where the neural network data will be stored, in this case `nndump.json`

`$ python cifar_model.py -o nndump.json`

Print the JSON file in the human readable format (with depth 3).

`$ python print_json.py nndump.json -d 3`


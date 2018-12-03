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

Running the model and collecting its data.
The `-o` option specifies where the NN data will be stored, in this case the `cifar4.json` or `sgemm.json` file. For more options run with the `-h` parameter.

Training the NN for image classification (CIFAR10 dataset)

`$ python cifar_model.py -o cifar4.json`

Training the model for regression using SGEMM dataset.

`$ python sgemm.py -o sgemm.json`

Print the JSON file in the human readable format (with depth 3).

`$ python print_json.py nndump.json -d 3`

Compare and visualize NNs

`$ python nnvis.py examples/sgemm-elu.json examples/sgemm-relu.json`

This should open a browser visualize the models in particular comparison of metrics, histograms, mean absolute differences and projection. Further, it saves the generated JavaScript in a file so you don't have to run the program again.

Producing outputs from convolutional layers. The outputs are saved in the `output` directory by default.

`$ python nnvis.py -i cifar4.json`
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
The `-o` option specifies where the NN data will be stored, in this case the `cifar4.json`/`sgemm.json` file. For more options run with the `-h` parameter.

Training the NN for image classification (CIFAR10 dataset)

`$ python cifar_model.py -o cifar4.json`

Training the model for regression using SGEMM dataset.

`$ python sgemm.py -o sgemm.json`

Print the JSON file in the human readable format (with depth 3).

`$ python print_json.py nndump.json -d 3`

Produce graphs and images from the collected data. The outputs are saved in the `out` directory by default.

`$ python nnvis.py cifar4.json`

The output directory will contain the following subdirectories:
 - learning_curve (metrics and losses)
 - histograms     (distribution of weights)
 - filters        (outputs of convolutional layers)
 - projection     (projection of validation data)
 - mean_abs_diff  (mean absolute difference of weights between epochs)

NNvis

Visualization of neural networks.

1. Requirements

Python >= 3.6


2. Install

$ pip install -r requirements.txt`


3. Usage

3.1 Running the model and collecting the data.

The -o option specifies where the NN data will be stored, in this case the
cifar4.json or sgemm.json file. For more options run with the -h parameter.

    3.1.1 Training the NN for image classification (CIFAR10 dataset)

    $ python cifar_model.py -o cifar4.json

    3.1.2 Training the model for regression using SGEMM dataset.

    $ python sgemm.py -o sgemm.json


    Note that, the datasets are downloaded automatically.

3.2 Print the JSON in a human readable format.

$ python print_json.py cifar4.json

3.3 Compare and visualize NNs

$ python nnvis.py examples/sgemm-elu.json examples/sgemm-relu.json

This should open a browser visualize the models in particular comparison of 
metrics, histograms, mean absolute differences and projection. Further, it 
saves the generated JavaScript in a file so you don't have to run the program 
again.

3.4 Produce and store graphs and images from the collected data.

$ python nnvis.py -i examples/cifar4.json

The result directory structure will look like this:
    output
        \- images              (input images)
        \- layer0 - {img}_{id} (outputs of layer0)
        \- layer1 - {img}_{id}
        \- ...

The produced images are outputs of convolutional networks. The first layer
should produce similar images to the input ones, but the output of other layers
is hard to interpret.
NNvis

Visualization of neural networks.


1. Requirements

Python >= 3.5


2. Install

$ pip install -r requirements.txt`


3. Usage

3.1 Running the model and collecting its data.
The -o option specifies where the NN data will be stored, in this case the cifar4.json/sgemm.json file. For more options run with the -h parameter.

    3.1.1 Training the NN for image classification (CIFAR10 dataset)

    $ python cifar_model.py -o cifar4.json

    3.1.2 Training the model for regression using SGEMM dataset.

    $ python sgemm.py -o sgemm.json

3.2 Print the json in a human readable format.

$ python print_json.py cifar4.json

3.3 Produce and store graphs and images from the collected data.

$ python nnvis cifar4.json

The result directory structure will look like this:
    out
       \- learning_curve (metrics and losses)
       \- histograms     (distribution of weights)
       \- filters
                  \- images              (input images)
                  \- layer0 - {img}_{id} (outputs of layer0)
                  \- layer1 - {img}_{id}
                  \- ...
       \- projection     (projection of validation data)
       \- mean_abs_diff  (mean absolute difference of weights between epochs)

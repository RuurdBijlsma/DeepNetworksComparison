# DeepNetworksComparison
Python script to run image classification using multiple different models, then output the performance metrics to csv files

## Getting the data

Make sure the folllowing packages are installed:
* pytorch
* torchvision
* cudatoolkit
* matplotlib
* scipy

Adjust `params.py` to adjust what is tested, currently the script trains 10 different models on the same dataset. The optimizer, loss function, batch size and epoch count can also be altered through this file.

Run `python test_network.py` to train the networks and export the data to the results folder

## Plotting the data

TODO

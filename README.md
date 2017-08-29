# Name Predictor
---
A small LSTM Neural Network to predict if a string is a name or not.

Dataset built from DBpedia and combinations of First Names and Last Names

### Prerequisites

* Python 3.6
* Numpy 1.12.1
* Pandas 0.20.2
* Scikit-Learn 0.18.2
* Seaborn 0.8*
* Matplotlib 2.0.2*
* Theano 0.9
* TensorFlow 1.2.1
* Jupyter Notebooks*

\*Only if wants to rerun Jupyter Notebooks

Miniconda Recommended to install the packages

### Data Preprocessing

The details about how to extract the data to build the dataset can be found in the file [Preprocessing.ipynb](https://github.com/omarch7/NamePredictor/blob/master/Preprocessing.ipynb) or the already processed dataset can downloaded from [data/full_names.tar.gz](https://github.com/omarch7/NamePredictor/raw/master/data/full_names.tar.gz) that contains 6 million samples.

### Training

Training can be found in [Training.ipynb](https://github.com/omarch7/NamePredictor/blob/master/Training.ipynb), the whole dataset took almost 3 hours on a Google Compute Engine instance with GPU K80 Tesla, for convenience the already trained model can be found in [models/](https://github.com/omarch7/NamePredictor/tree/master/models).

The model achieved 96% of accuracy.

### Prediction

#### Jupyter Notebooks

The prediction can be done using [Predicting.ipynb](https://github.com/omarch7/NamePredictor/blob/master/Predicting.ipynb) some samples were added to test the model manually.

#### Command Line

To predict if a strings contain a person name can be done using the command line, the program takes as input a tab separated file [input.tsv](https://github.com/omarch7/NamePredictor/blob/master/input.tsv) and generates an output with the same format but with the correct label, optionally probabilities can be added as well.

```shell
python NamePredictor.py [model] [input] [output]
```
Example
```shell
python NamePredictor.py models/model.h5 input.tsv output.tsv
```
To add verbosity
```shell
python NamePredictor.py models/model.h5 input.tsv output.tsv --verbosity
```
To add probabilities to the output
```shell
python NamePredictor.py models/model.h5 input.tsv output.tsv --probabilities
```

---

Developed by Omar Contreras [omarch7@gmail.com](mailto:omarch7@gmail.com)

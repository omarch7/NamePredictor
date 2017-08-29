import sys
import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


class NamePredictor:
    def __init__(self, model_path: str = None, verbosity=False):
        self.verbosity = verbosity
        if not model_path:
            print("Need to specify model_path")
            sys.exit()
        if verbosity:
            print("Loading Model...")
        self.model = load_model(model_path)
        if verbosity:
            print("Model loaded!")

    @staticmethod
    def encode_string(s):
        encoded = []
        for c in s:
            idx = ord(c)
            if idx >= 32 and idx <= 126:
                encoded.append(idx-31)
            elif idx > 126:
                # Rare Characters like accented letters and specific language characters
                encoded.append(96)
        return encoded

    def encode_input(self, data: np.ndarray):
        return pad_sequences(list(map(lambda s: self.encode_string(s), data)), maxlen=50, dtype=np.int32, padding="post")

    def predict_file(self, input_file: str, output_file: str, prob: bool = False):
        if self.verbosity:
            print("Reading {}...".format(input_file))
        strings = pd.read_csv(input_file, index_col=False, sep="\t")
        if self.verbosity:
            print("Encoding data...")
        x = self.encode_input(strings['string'].values)
        if self.verbosity:
            print("Predicting {} samples...".format(x.shape[0]))
        predictions = self.model.predict(x)
        if prob:
            strings['probabilities'] = predictions
        strings['is_person_name'] = np.apply_along_axis(lambda s: True if s > 0.5 else False, 1, predictions)
        if self.verbosity:
            print("Saving {}...".format(output_file))
        strings.to_csv(output_file, index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path for the trained h5 model")
    parser.add_argument("input", help="Path for the input in tsv")
    parser.add_argument("output", help="Path for the output in tsv")
    parser.add_argument("--verbosity", action="store_true", help="Print messages")
    parser.add_argument("--probabilities", action="store_true", help="Add probabilities to output file")

    args = parser.parse_args()
    print("Welcome to Name Predictor")
    name_predictor = NamePredictor(args.model, args.verbosity)
    name_predictor.predict_file(args.input, args.output, args.probabilities)
    print("Thanks for using\nOmar Contreras")

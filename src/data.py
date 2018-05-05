import numpy as np
import csv
import re
import os

class EssayData:
    def __init__(self, test):
        """
        Initialize training/testing data

        Args:
            test (boolean): specifies to pick data from training/testing
        """

        self.test = test
        if test:
            self.X, self.P, self.files  = self.load_essays('../input/testing/')
        else:
            self.X, self.P, self.Y, self.files  = self.load_essays('../input/training/')


    def load_essays(self, filename):
        """
        Load data from file

        Args:
            filename (str): parent directory of .csv file

        Returns:
            X (list): list of written essays
            P (list): list of topic/prompt
            Y (list): list of labels (if test=True)
            files (list): list of file names
        """

        X = []; P = []; Y = []; files = []
        label_dict = {}

        with open(filename + 'index.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                prompt = re.search('\t\t(.*)\t\t', row['prompt']).group(1)
                if self.test:
                    label_dict[row['filename']] = [prompt]
                else:
                    label_dict[row['filename']] = [prompt, row['grade']]

        for essay_file in os.listdir(filename + 'essays/'):
            essay = open(filename + '/essays/' + essay_file, 'r').read()
            X.append(essay)
            P.append(label_dict[essay_file][0])
            if not self.test:
                Y.append(1 if label_dict[essay_file][1] == 'high' else -1)
            files.append(essay_file)
        if self.test:
            return np.array(X), np.array(P), files
        else:
            return np.array(X), np.array(P), np.array(Y), files

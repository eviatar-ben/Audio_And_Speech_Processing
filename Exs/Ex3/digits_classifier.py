import math
import os
from abc import abstractmethod
import numpy as np
import torch
import typing as tp
from dataclasses import dataclass
import librosa
from numpy.linalg import norm


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    # we will use this to give an absolute path to the data, make sure you read the data using this argument. 
    # you may assume the train data is the same
    path_to_training_data_dir: str = r".\train_files"
    path_to_test_data_dir: str = r".\test_files"

    # you may add other args here


class DigitClassifier():
    """
    You should Implement your classifier object here
    """

    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir
        self.path_to_test_data_dir = args.path_to_test_data_dir

    @staticmethod
    def dtw(x, y, dist=lambda x, y: norm(x - y, ord=2)):
        from numpy import array, zeros, argmin, inf, equal, ndim
        from scipy.spatial.distance import cdist
        """
        Computes Dynamic Time Warping (DTW) of two sequences.

        :param array x: N1*M array
        :param array y: N2*M array
        :param func dist: distance used as cost measure

        Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
        """

        def _traceback(D):
            i, j = array(D.shape) - 2
            p, q = [i], [j]
            while ((i > 0) or (j > 0)):
                tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
                if (tb == 0):
                    i -= 1
                    j -= 1
                elif (tb == 1):
                    i -= 1
                else:  # (tb == 2):
                    j -= 1
                p.insert(0, i)
                q.insert(0, j)
            return array(p), array(q)

        r, c = len(x), len(y)
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
        D1 = D0[1:, 1:]  # view

        for i in range(r):
            for j in range(c):
                D1[i, j] = dist(x[i], y[j])

        C = D1.copy()

        for i in range(r):
            for j in range(c):
                D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
        if len(x) == 1:
            path = zeros(len(y)), range(len(y))
        elif len(y) == 1:
            path = range(len(x)), zeros(len(x))
        else:
            path = _traceback(D0)
        return D1[-1, -1] / sum(D1.shape), C, D1, path

    @staticmethod
    def dtw1(x, y):
        # cost_matrix = scipy.spatial.distance.cdist(x, y, metric='seuclidean')
        cost_matrix = torch.cdist(x, y)
        m, n = np.shape(cost_matrix)
        for i in range(m):
            for j in range(n):
                if (i == 0) & (j == 0):
                    cost_matrix[i, j] = cost_matrix[i, j]
                elif i == 0:
                    cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i, j - 1]

                elif j == 0:
                    cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i - 1, j]

                else:
                    min_local_dist = cost_matrix[i - 1, j]

                    if min_local_dist > cost_matrix[i, j - 1]:
                        min_local_dist = cost_matrix[i, j - 1]

                    if min_local_dist > cost_matrix[i - 1, j - 1]:
                        min_local_dist = cost_matrix[i - 1, j - 1]

                    cost_matrix[i, j] = cost_matrix[i, j] + min_local_dist
        return cost_matrix, cost_matrix[m - 1, n - 1]

    def get_train_data(self, validate=True):
        """
        function to get the training data
        return: list of training data
        """
        # open files in path_to_training_data
        sub_directories = ["one", "two", "three", "four", "five"]
        train_data_mfcc = []
        train_data_labels = []
        for i, sub in enumerate(sub_directories):
            validation = [f for f in os.listdir(os.path.join(self.path_to_training_data, sub)) if f.endswith(".wav")]
            if validate:
                validation = validation[:int(len(validation) * 0.8)]
            for file in validation:
                y, sr = librosa.load(os.path.join(self.path_to_training_data, sub, file))
                mfcc = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
                # mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
                # mfcc /= np.std(mfcc, axis=0)

                train_data_mfcc.append(mfcc)
                train_data_labels.append(i + 1)

        return train_data_mfcc, train_data_labels

    def knn(self, audio_files: tp.Union[tp.List[str], torch.Tensor], euclidean=False, librosa_dwt=False) -> tp.List[
        int]:
        x, y = self.get_train_data()
        result = []

        for f_path in audio_files:
            wav, sr = librosa.load(f_path)
            mfcc = librosa.feature.mfcc(y=wav, sr=sr)
            # mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
            # mfcc /= np.std(mfcc, axis=0)

            nearest_neighbor = None
            nearest_neighbor_distance = math.inf
            for i, n in enumerate(x):
                if euclidean:
                    # euclidean distance
                    distance = torch.dist(torch.tensor(mfcc).flatten(), torch.tensor(n).flatten())
                # DTW distance
                elif librosa_dwt:
                    D, wp = librosa.sequence.dtw(mfcc, n)
                    # get the last element of the last row
                    distance = D[-1, -1]  # todo: check this out
                else:
                    distance, cost, acc_cost, path = DigitClassifier.dtw(torch.tensor(mfcc), torch.tensor(n))
                if distance.item() < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance.item()
                    nearest_neighbor = y[i]
            result.append(nearest_neighbor)

        return result

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        return self.knn(audio_files, euclidean=True)

    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor], librosa_dwt=False) -> \
            tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        return self.knn(audio_files, librosa_dwt=librosa_dwt)

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        # todo : check if both distance needed

        audio_files_full_path = [os.path.join(self.path_to_test_data_dir, f) for f in audio_files]
        dwt_labels = self.classify_using_DTW_distance(audio_files_full_path, librosa_dwt=False)
        euclidean_labels = self.classify_using_eucledian_distance(audio_files_full_path)

        result = []
        for i, f in enumerate(audio_files):
            result.append(f"{f} - {euclidean_labels[i]} - {dwt_labels[i]}")

        with open("output.txt", "w") as f:
            for i in result:
                f.write(i)
                f.write("\n")
        return result


class ClassifierHandler:
    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        return DigitClassifier(ClassifierArgs())
        # raise NotImplementedError("function is not implemented")


def cross_validate():
    classifier = DigitClassifier(ClassifierArgs())
    sub_directories = ["one", "two", "three", "four", "five"]
    sanity_check = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

    total_librosa_dwt_accuracy = 0
    total_local_dwt_accuracy = 0
    total_euclidean_accuracy = 0
    total_labels = 0

    for i, sub in enumerate(sub_directories):
        audio_files = [f for f in os.listdir(os.path.join(classifier.path_to_test_data_dir, sub)) if f.endswith(".wav")]
        audio_files = [os.path.join(classifier.path_to_test_data_dir, sub, f) for f in audio_files]

        librosa_dwt_labels = classifier.classify_using_DTW_distance(audio_files, librosa_dwt=True)
        dwt_labels = classifier.classify_using_DTW_distance(audio_files, librosa_dwt=False)
        euclidean_labels = classifier.classify_using_eucledian_distance(audio_files)

        assert len(librosa_dwt_labels) == len(dwt_labels) == len(euclidean_labels) == len(audio_files)
        assert sanity_check[sub] == i + 1

        print(f"sub directory: {sub}")

        print(f"dwt_labels: {dwt_labels}")
        print(f"librosa_dwt_labels: {librosa_dwt_labels}")
        print(f"euclidean_labels: {euclidean_labels}")

        print(f"compatibility between librosa dwt and local dwt:"
              f" {sum([l1 == l2 for l1, l2 in zip(librosa_dwt_labels, dwt_labels)]) / len(librosa_dwt_labels)}")
        print(f"compatibility between euclidean and local dwt:"
              f" {sum([l1 == l2 for l1, l2 in zip(euclidean_labels, dwt_labels)]) / len(euclidean_labels)}")

        local_dwt_correct_label = sum([l == i + 1 for l in librosa_dwt_labels])
        librosa_dwt_correct_label = sum([l == i + 1 for l in dwt_labels])
        euclidean_correct_label = sum([l == i + 1 for l in euclidean_labels])

        print(f"Accuracy local dwt: {local_dwt_correct_label / len(librosa_dwt_labels)}")
        print(f"Accuracy librosa dwt: {librosa_dwt_correct_label / len(dwt_labels)}")
        print(f"Accuracy euclidean: {euclidean_correct_label / len(euclidean_labels)}")
        print(f"-----------------------")

        total_local_dwt_accuracy += local_dwt_correct_label
        total_librosa_dwt_accuracy += librosa_dwt_correct_label
        total_euclidean_accuracy += euclidean_correct_label

        total_labels += len(librosa_dwt_labels)

    print(f"Total Accuracy local dwt: {total_local_dwt_accuracy / total_labels}")
    print(f"Total Accuracy librosa dwt: {total_librosa_dwt_accuracy / total_labels}")
    print(f"Total Accuracy euclidean: {total_euclidean_accuracy / total_labels}")


def basic_test():
    classifier = DigitClassifier(ClassifierArgs())
    sub_directories = ["one", "two", "three", "four", "five"]
    audio_files = []
    for i, sub in enumerate(sub_directories):
        audio_files_in_sub_dir = [f for f in os.listdir(os.path.join(classifier.path_to_test_data_dir, sub)) if
                                  f.endswith(".wav")]
        audio_files += [os.path.join(sub, f) for f in audio_files_in_sub_dir]

    classifier = ClassifierHandler.get_pretrained_model()
    classifier.classify(audio_files)


if __name__ == '__main__':
    cross_validate()
    # basic_test()

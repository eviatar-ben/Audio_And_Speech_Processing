from abc import abstractmethod
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import librosa
import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

SR = 22050
MFCC_FEATURES_NUMBER = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with
    default values (so run won't break when we test this).
    """
    batch_size: int = 32
    num_epochs: int = 80
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json"  # you should use this file path to load your test data
    # other training hyper parameters


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.001


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        self.paths = []
        self.labels = []
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            self.paths.append(item['path'])
            self.labels.append(item['label'])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if self.labels[index] == 'classical':
            label = 0
        elif self.labels[index] == 'heavy-rock':
            label = 1
        else:
            label = 2
        wav, sr = librosa.load(path, sr=SR)
        return wav, label


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters = OptimizationParameters(), **kwargs):
        """
        This defines the classifier object.
        - You should defiend your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        self.labels_num = len(Genre)
        self.regularization = 0.33
        self.feat_num = MFCC_FEATURES_NUMBER * 3
        self.weights = torch.randn((self.feat_num,1), dtype=torch.double, device=device)
        self.weights = self.weights.repeat((1, self.labels_num))
        self.biases = torch.randn(1, device=device)
        self.biases = self.biases.repeat((self.labels_num))
        self.batch_size = kwargs['batch_size']
        self.learning_rate = opt_params.learning_rate
        self.dot = torch.zeros(self.batch_size, self.labels_num, device=device)  # dot product of weights and features
        self.sig_result = torch.zeros(self.batch_size, self.labels_num, device=device)  # sigmoid result

    def exctract_feats(self, wavs, plot=False, mean_normalization_mfcc=True, mean_normalization_f=True,
                       mean_normalization_delta=True, mean_normalization_delta2=True):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        fs = np.zeros((len(wavs), MFCC_FEATURES_NUMBER))
        delta = np.zeros((len(wavs), MFCC_FEATURES_NUMBER))
        delta2 = np.zeros((len(wavs), MFCC_FEATURES_NUMBER))
        # since mfcc support multi-channel but not multi audio files- for loop is needed
        # as mentioned here https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
        # to balance the spectrum and improve the Signal-to-Noise (SNR),
        # we can simply subtract the mean of each coefficient from all frames.
        # mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
        wavs = wavs.numpy()
        for i, wav in enumerate(wavs):
            mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=MFCC_FEATURES_NUMBER, dct_type=2)
            if mean_normalization_mfcc:
                mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
                mfcc /= np.std(mfcc, axis=0)
            f = np.mean(mfcc, axis=1)

            if mean_normalization_f:
                f -= np.mean(f)
                f /= np.std(f)
            fs[i, :] = f

            # delta
            delta[i, :] = np.mean(librosa.feature.delta(mfcc), axis=1)
            if mean_normalization_delta:
                delta[i, :] -= np.mean(delta[i, :])
                # delta[i, :] /= np.std(delta[i, :])
            if mean_normalization_delta2:
                delta2[i, :] -= np.mean(delta2[i, :])
                # delta2[i, :] /= np.std(delta2[i, :])
            delta2[i, :] = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)
            if plot:
                self.plot_mfcc(mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2))

        features = np.hstack((fs, delta, delta2))

        return torch.tensor(features, device=device)

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        # for i in range(self.labels_num):
        #     self.dot[i] = torch.mm(feats, self.weights[i]) + self.biases[i]
        #     self.sig_result[i] = torch.sigmoid(self.dot[i])
        self.dot = torch.mm(feats, self.weights) + self.biases
        self.sig_result = torch.sigmoid(self.dot)

    def backward(self, feats: torch.Tensor, labels: torch.Tensor, requires_grad: bool = True):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence
        OptimizationParameters are passed to the initialization function
        """

        labels = labels.to(device)
        cur_batch_size = feats.shape[0]
        label0 = (labels == 0).double()
        label1 = (labels == 1).double()
        label2 = (labels == 2).double()
        labels_norm = torch.stack((label0, label1, label2), dim=1)
        dC_dW = torch.mm(feats.T, self.sig_result - labels_norm) / cur_batch_size + self.weights * self.regularization
        dC_dB = torch.sum(self.sig_result - labels_norm, dim=0) / cur_batch_size + self.biases * self.regularization
        if requires_grad:
            self.weights = self.weights - self.learning_rate * dC_dW
            self.biases = self.biases - self.learning_rate * dC_dB
        loss = [torch.nn.functional.binary_cross_entropy(self.sig_result[:, i].view(cur_batch_size, 1),
                                                         labels_norm[:, i].view(cur_batch_size, 1)) +
                self.regularization * torch.sum(self.weights ** 2) / 2 + self.regularization * torch.sum(self.biases ** 2) / 2 for i in range(self.labels_num)]

        return loss


    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object,
        should return a tuple: (weights, biases)
        """
        return (self.weights, self.biases)

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor)
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        # extract features from wavs
        feats = self.exctract_feats(wavs)
        # forward pass
        self.forward(feats)
        # return predicted labels
        return torch.argmax(self.sig_result, dim=1)


    @staticmethod
    def plot_mfcc(mfcc, mfcc_delta, mfcc_delta2):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
        img1 = librosa.display.specshow(mfcc, ax=ax[0], x_axis='time')
        ax[0].set(title='MFCC')
        ax[0].label_outer()
        img2 = librosa.display.specshow(mfcc_delta, ax=ax[1], x_axis='time')
        ax[1].set(title=r'MFCC-$\Delta$')
        ax[1].label_outer()
        img3 = librosa.display.specshow(mfcc_delta2, ax=ax[2], x_axis='time')
        ax[2].set(title=r'MFCC-$\Delta^2$')
        fig.colorbar(img1, ax=[ax[0]])
        fig.colorbar(img2, ax=[ax[1]])
        fig.colorbar(img3, ax=[ax[2]])
        fig.show()


class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters = TrainingParameters()) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        model = MusicClassifier(OptimizationParameters(), batch_size=training_parameters.batch_size)
        # train model
        dataset = MusicDataset(training_parameters.train_json_path)
        data_generator = torch.utils.data.DataLoader(dataset, batch_size=training_parameters.batch_size, shuffle=True)
        train_losses = []
        test_dataset = MusicDataset(training_parameters.test_json_path)
        best_model = None
        best_acc = 0
        test_losses = []
        for epoch in range(training_parameters.num_epochs):
            with tqdm (data_generator, desc=f'Epoch {epoch + 1}/{training_parameters.num_epochs}', unit='batch') as t:
                for X, y in t:
                    feats = model.exctract_feats(X)
                    # forward pass
                    model.forward(feats)
                    # backward pass
                    loss = model.backward(feats, y.type(torch.double))
                    train_losses.append([loss[i].cpu().detach().numpy() for i in range(len(loss))])
                    t.set_postfix(loss=loss)
            # test model on the test set, compute the accuracy:
            test_data_generator = torch.utils.data.DataLoader(test_dataset, batch_size=training_parameters.batch_size, shuffle=True)
            test_acc = 0
            for X, y in test_data_generator:
                # compute accuracy
                test_acc += torch.sum(model.classify(X) == y.to(device))
                # # compute test loss
                # feats = model.exctract_feats(X)
                # model.forward(feats)
                # loss = model.backward(feats, y.type(torch.double), requires_grad=False)
                # test_losses.append([loss[i].cpu().detach().numpy() for i in range(len(loss))])
            test_acc = test_acc.item()
            test_acc /= len(test_dataset)
            print(f'test accuracy: {test_acc}')

            torch.save(model.get_weights_and_biases(), f'model_epoch={epoch+1}.pt')
        # save best model

        model1_losses = [train_losses[i][0] for i in range(len(train_losses))]
        model2_losses = [train_losses[i][1] for i in range(len(train_losses))]
        model3_losses = [train_losses[i][2] for i in range(len(train_losses))]
        # plot all losses together
        plt.plot(model1_losses, label='model1')
        plt.plot(model2_losses, label='model2')
        plt.plot(model3_losses, label='model3')
        plt.title('train losses')
        plt.legend()
        plt.show()
        # plot test losses
        model1_losses = [test_losses[i][0] for i in range(len(test_losses))]
        model2_losses = [test_losses[i][1] for i in range(len(test_losses))]
        model3_losses = [test_losses[i][2] for i in range(len(test_losses))]
        # plot all losses together
        plt.plot(model1_losses, label='model1')
        plt.plot(model2_losses, label='model2')
        plt.plot(model3_losses, label='model3')
        plt.title('test losses')
        plt.legend()
        plt.show()

        return model

    @staticmethod
    def save_model(model: MusicClassifier):
        """
        This function saves the model to the given path
        """
        # save weights and biases
        torch.save(model.get_weights_and_biases(), 'model.pt')

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights /
        hyperparameters and return the loaded model
        """
        model = MusicClassifier(OptimizationParameters(), batch_size=TrainingParameters.batch_size)
        model.weights, model.biases = torch.load('model.pt')
        return model

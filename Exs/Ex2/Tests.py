import librosa
import torch

import genre_classifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def test_feature_extraction():
    """
    This function tests the feature extraction method.
    """
    wav, sr = librosa.load(
        r"C:\Users\eviatar\PycharmProjects\Audio And Speech Processing\Exs\Ex2\parsed_data\reggae\test\2.mp3",
        sr=None)
    classifier = genre_classifier.MusicClassifier(batch_size=1)
    classifier.exctract_feats(torch.Tensor([wav]), plot=True, mean_normalization_mfcc=False,
                              mean_normalization_delta=False, mean_normalization_delta2=False)


def main():
    test_feature_extraction()
    train_dataset = genre_classifier.MusicDataset("jsons/train.json")
    test_dataset = genre_classifier.MusicDataset("jsons/test.json")
    model = genre_classifier.ClassifierHandler.train_new_model()

    data_generator = torch.utils.data.DataLoader(train_dataset, batch_size=34)
    losses = []
    for epoch in range(10):
        for X, y in data_generator:
            feats = model.exctract_feats(X)
            # forward pass
            model.forward(feats)
            # backward pass
            loss = model.backward(feats, y.type(torch.double))
            print(loss)
            losses.append(loss)
    model = handler.train_new_model()

    logistic_classifier = linear_model.logistic.LogisticRegression()
    logistic_classifier.fit(X_train, y_train)
    logistic_predictions = logistic_classifier.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    logistic_cm = confusion_matrix(y_test, logistic_predictions)

    print("logistic accuracy = " + str(logistic_accuracy))
    print("logistic_cm:")
    print(logistic_cm)


if __name__ == '__main__':
    main()

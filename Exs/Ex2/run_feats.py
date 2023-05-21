import numpy as np

import genre_classifier
from sklearn.linear_model import SGDClassifier
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import wandb
from genre_classifier import MFCC_FEATURES_NUMBER

# WB = True
WB = False
LR = 1e-3
EPOCHS = 1
RUN = f"Everything is normalized {EPOCHS} epochs"
GROUP = "MFCC DELTA DELTA2"


def init_w_and_b(group, project, d="", description="", run="", LR=1e-3, EPOCHS=10, architecture="GAN"):
    wandb.init(
        # settings=wandb.Settings(start_method="fork"),
        # Set the project where this run will be logged
        group=group,
        project=project,
        name=f"{description}{run} {d} MFCC features",
        notes='',
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": architecture,
            "dataset": "MNIST",
            "epochs": EPOCHS,
        })


if __name__ == "__main__":
    if WB:
        init_w_and_b(project="Genre Classifier ", group=GROUP,
                     d=str(MFCC_FEATURES_NUMBER), description="", run=RUN, LR=LR,
                     EPOCHS=EPOCHS, architecture="GAN")
    dataset = genre_classifier.MusicDataset(genre_classifier.TrainingParameters.train_json_path)
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=genre_classifier.TrainingParameters.batch_size,
                                                 shuffle=True)
    model = genre_classifier.MusicClassifier(genre_classifier.OptimizationParameters(),
                                             batch_size=genre_classifier.TrainingParameters.batch_size)
    lr = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=0)
    for epoch in range(EPOCHS):
        feats = 0
        for X, y in data_generator:
            feats = model.exctract_feats(X)
            lr.partial_fit(feats, y, classes=np.unique([0, 1, 2]))
            # print(lr.score(feats, y))
        if WB:
            wandb.log({"Train Accuracy": lr.score(feats, y)}, step=epoch)
    # test the model:
    test_data = genre_classifier.MusicDataset(genre_classifier.TrainingParameters.test_json_path)
    test_data_generator = torch.utils.data.DataLoader(test_data,
                                                      batch_size=genre_classifier.TrainingParameters.batch_size,
                                                      shuffle=False)

    print("test data:")
    mean_accuracy = 0
    for X, y in test_data_generator:
        feats = model.exctract_feats(X)
        predictions = lr.predict(feats)
        for i in range(len(predictions)):
            print(f"prediction: {predictions[i]}, label: {y[i]}")
        print(lr.score(feats, y))

        logistic_accuracy = accuracy_score(y, predictions)
        mean_accuracy += logistic_accuracy
        # logistic_cm = confusion_matrix(y, predictions)

        print("logistic accuracy = " + str(logistic_accuracy))
        print("logistic_cm:")
        # print(logistic_cm)
    if WB:
        wandb.log({"Test mean Accuracy": mean_accuracy / len(test_data_generator)})
        # wandb.log({"Test Confusion Matrix": logistic_cm})

import numpy as np

import genre_classifier
import torch
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    model = genre_classifier.ClassifierHandler.train_new_model()
    testdata = genre_classifier.MusicDataset(genre_classifier.TrainingParameters.test_json_path)
    test_data_generator = torch.utils.data.DataLoader(testdata,
                                                      batch_size=genre_classifier.TrainingParameters.batch_size)
    genre_classifier.ClassifierHandler.save_model(model)

    print("test data:")
    test_acc = []
    for X, y in test_data_generator:
        predictions = model.classify(X)
        for i in range(len(predictions)):
            print(f"prediction: {predictions[i]}, label: {y[i]}")
        test_acc.append(accuracy_score(y, predictions.cpu().numpy()))

    test_acc = np.asarray(test_acc)
    print(f"Mean acc:{np.mean(test_acc)}")
    print(f"Variance acc:{np.var(test_acc)}")

import pandas as pd


def submit(model, test_dataset, test_data_id):
    test_preds = model.predict(test_dataset)
    submission = pd.DataFrame(
        {'id': test_data_id, 'target': test_preds.flatten()})
    submission = submission.set_index('id')
    submission.to_csv("submission.csv")

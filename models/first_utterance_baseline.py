import collections
import json
import os
from model_evaluation import combine_dataset, combine_predictions, evaluate

parent = os.path.join(os.getcwd(), os.pardir)
data_folder = os.path.join(parent, 'Data')
all_data_for_model_folder = os.path.join(data_folder, '1_pre_only_switch_speaker')


def first_utterance_baseline(eval_dataset):
    preds = collections.OrderedDict()
    for sample in eval_dataset:
        preds[sample['id']] = sample['context'][:sample['context'].find('&', 1) - 1]

    return preds


def main():
    with open(os.path.join(all_data_for_model_folder, f'validation_data.json'), "r", encoding='utf-8') as reader:
        validation_data = json.load(reader)["data"]
    with open(os.path.join(all_data_for_model_folder, f'test_data.json'), "r", encoding='utf-8') as reader:
        test_data = json.load(reader)["data"]

    validation_predictions = first_utterance_baseline(validation_data)
    test_predictions = first_utterance_baseline(test_data)

    new_validation_predictions = combine_predictions(validation_predictions)
    new_validation_data = combine_dataset(validation_data)
    result_prediction_statistics_dict = evaluate(new_validation_data, new_validation_predictions)

    new_test_predictions = combine_predictions(test_predictions)
    new_test_data = combine_dataset(test_data)
    result_prediction_statistics_dict = evaluate(new_test_data, new_test_predictions, data_type='test')


if __name__ == '__main__':
    main()

import os
import json
from model_evaluation import combine_dataset, evaluate

parent = os.path.join(os.getcwd(), os.pardir)
data_folder = os.path.join(parent, 'Data')
all_data_for_model_folder = os.path.join(data_folder, '1_pre_only_switch_speaker')


def main():
    for data_type in ['validation', 'test']:
        with open(os.path.join(all_data_for_model_folder, f'{data_type}_data.json'), "r", encoding='utf-8') as reader:
            test_data = json.load(reader)["data"]

        new_test_data = combine_dataset(test_data)
        result_prediction_statistics_dict = evaluate(new_test_data, data_type=data_type, humans=True)


if __name__ == '__main__':
    main()

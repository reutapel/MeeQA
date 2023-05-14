import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import datetime
from hyper_parameter_tunning import run_model

parent = os.path.join(os.getcwd(), os.pardir)
data_folder = os.path.join(parent, 'Data')
models_results_folder = os.path.join(parent, 'model_results')
if not os.path.exists(models_results_folder):
    os.mkdir(models_results_folder)


eval_grid_search_params = [
    {'has_answer_threshold': has_answer_threshold,
     'null_score_diff_threshold': null_score_diff_threshold,
     'max_answer_length': max_answer_length
     }
    for has_answer_threshold in [0.6, 0.7]
    for null_score_diff_threshold in [0.1, 0.2]
    for max_answer_length in [200, 250]
]

ablation_analysis_model_grid_search_params = {
    'no_has_answer_loss_weight': [
        {'has_answer_loss_weight': 0,
         'start_end_prediction_loss_weight': start_end_prediction_loss_weight,
         'start_end_labels_loss_weight': start_end_labels_loss_weight,
         'learning_rate': 3e-05}
        for start_end_prediction_loss_weight in [0.2, 0.3]
        for start_end_labels_loss_weight in [0.8, 0.9]
    ],
    'no_start_end_prediction_loss_weight': [
        {'has_answer_loss_weight': has_answer_loss_weight,
         'start_end_prediction_loss_weight': 0,
         'start_end_labels_loss_weight': start_end_labels_loss_weight,
         'learning_rate': 3e-05}
        for has_answer_loss_weight in [0.7, 0.8]
        for start_end_labels_loss_weight in [0.8, 0.9]
    ],
    'no_start_end_labels_loss_weight': [
        {'has_answer_loss_weight': has_answer_loss_weight,
         'start_end_prediction_loss_weight': start_end_prediction_loss_weight,
         'start_end_labels_loss_weight': 0,
         'learning_rate': 3e-05}
        for has_answer_loss_weight in [0.7, 0.8]
        for start_end_prediction_loss_weight in [0.2, 0.3]
        ],
    }


def main():
    print("Ablation Analysis")
    print(f"{os.getcwd()}")
    try:
        print(subprocess.check_call('nvidia-smi'))
    except BaseException as e:
        print("no GPU available")

    data_file_folders = [os.path.join(data_folder, '1_pre_only_switch_speaker')]
    models = [['albert-base-v2', 'albert'], ['microsoft/deberta-base', 'deberta'], ['bert-base-uncased', 'bert']]
    for model_name_or_path, model_name in models:
        for data_file_folder in data_file_folders:
            data_type = data_file_folder.split('/')[-1]
            train_file = f'{data_file_folder}/train_data.json'
            validation_file = f'{data_file_folder}/validation_data.json'
            test_file = f'{data_file_folder}/test_data.json'
            for removed_part, model_grid_search_params in ablation_analysis_model_grid_search_params.items():
                print(f'Remove {removed_part} from loss')
                for i, model_param_dict in enumerate(model_grid_search_params):
                    output_dir = os.path.join(models_results_folder,
                                              f'start_end_has_answer_use_labels_predictions_'
                                              f'{data_type}_{model_name}_{removed_part}_run_id_{i}')
                    if os.path.exists(output_dir):
                        print(f'output_dir {output_dir} exists --> continue')
                        continue
                    cache_dir = f'{output_dir}/cache'
                    has_answer_loss_weight = model_param_dict['has_answer_loss_weight']
                    start_end_prediction_loss_weight = model_param_dict['start_end_prediction_loss_weight']
                    start_end_labels_loss_weight = model_param_dict['start_end_labels_loss_weight']
                    learning_rate = model_param_dict['learning_rate']

                    for j, eval_param_dict in enumerate(eval_grid_search_params):
                        print(f'{datetime.datetime.now()}: Start version {j} of model {i}: {output_dir}')
                        max_answer_length = eval_param_dict['max_answer_length']
                        null_score_diff_threshold = eval_param_dict['null_score_diff_threshold']
                        has_answer_threshold = eval_param_dict['has_answer_threshold']

                        run_model(model_name_or_path, train_file, validation_file, test_file, cache_dir, output_dir,
                                  has_answer_threshold, null_score_diff_threshold, has_answer_loss_weight,
                                  start_end_prediction_loss_weight, start_end_labels_loss_weight,
                                  learning_rate=learning_rate, max_answer_length=max_answer_length)


if __name__ == '__main__':
    main()

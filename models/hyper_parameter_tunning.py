import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import datetime
import run_qa_has_answer

parent = os.path.join(os.getcwd(), os.pardir)
data_folder = os.path.join(parent, 'Data')
models_results_folder = os.path.join(parent, 'model_results')
if not os.path.exists(models_results_folder):
    os.mkdir(models_results_folder)


model_grid_search_params = [
    {'has_answer_loss_weight': has_answer_loss_weight,
     'start_end_prediction_loss_weight': start_end_prediction_loss_weight,
     'start_end_labels_loss_weight': start_end_labels_loss_weight,
     'learning_rate': learning_rate}
    for has_answer_loss_weight in [0.7, 0.8]
    for start_end_prediction_loss_weight in [0.2, 0.3]
    for start_end_labels_loss_weight in [0.8, 0.9]
    for learning_rate in [3e-05]
]

eval_grid_search_params = [
    {'has_answer_threshold': has_answer_threshold,
     'null_score_diff_threshold': null_score_diff_threshold,
     'max_answer_length': max_answer_length
     }
    for has_answer_threshold in [0.6, 0.7]
    for null_score_diff_threshold in [0.1, 0.2]
    for max_answer_length in [200, 250]
]


def run_model(model_name_or_path, train_file, validation_file, test_file, cache_dir, output_dir,
              has_answer_threshold=0.0, null_score_diff_threshold=0.0, has_answer_loss_weight=0.0,
              start_end_prediction_loss_weight=0.0, start_end_labels_loss_weight=0.0, max_seq_length=512,
              first_feature_only=True, doc_stride=64, version_2_with_negative=True,
              learning_rate=5e-05, max_answer_length=200, num_train_epochs=2, per_device_train_batch_size=8,
              per_device_eval_batch_size=8,  do_predict=True,  dataset_name=None,
              start_end_has_answer_use_predictions_has_answer_in_loss=True, use_original_start_end=False):

    command_to_run = [f"--model_name_or_path", model_name_or_path, f"--cache_dir", cache_dir, f"--output_dir",
                      output_dir, f"--train_file", train_file, f"--learning_rate", str(learning_rate),
                      f"--validation_file", validation_file, f"--test_file", test_file,
                      f"--max_seq_length", str(max_seq_length), f"--do_train", f"--do_eval",
                      f'--do_predict', str(do_predict), f"--first_feature_only", str(first_feature_only),
                      f"--doc_stride", str(doc_stride), f"--version_2_with_negative", str(version_2_with_negative),
                      f"--max_answer_length", str(max_answer_length),
                      f"--num_train_epochs", str(num_train_epochs),
                      f"--per_device_train_batch_size", str(per_device_train_batch_size),
                      f"--per_device_eval_batch_size", str(per_device_eval_batch_size),
                      f"--start_end_has_answer_use_predictions_has_answer_in_loss",
                      str(start_end_has_answer_use_predictions_has_answer_in_loss),
                      f"--use_original_start_end", str(use_original_start_end),
                      f"--has_answer_threshold", str(has_answer_threshold),
                      f"--null_score_diff_threshold", str(null_score_diff_threshold),
                      f"--has_answer_loss_weight", str(has_answer_loss_weight),
                      f"--start_end_prediction_loss_weight", str(start_end_prediction_loss_weight),
                      f"--start_end_labels_loss_weight", str(start_end_labels_loss_weight)]
    if dataset_name is not None:
        command_to_run.append(f"--dataset_name")
        command_to_run.append(dataset_name)
    run_qa_has_answer.main(command_to_run)


def main():
    print("Getting Started")
    print(f"{os.getcwd()}")
    try:
        print(subprocess.check_call('nvidia-smi'))
    except BaseException as e:
        print("no GPU available")

    data_file_folders =\
        [
            f'{data_folder}/1_pre_only_switch_speaker',
            f'{data_folder}/2_pre_only_switch_speaker',
            f'{data_folder}/only_switch_speaker',
            f'{data_folder}/original'
        ]
    models = [['albert-base-v2', 'albert'], ['bert-base-uncased', 'bert'], ['microsoft/deberta-base', 'deberta']]
    for model_name_or_path, model_name in models:
        for data_file_folder in data_file_folders:
            data_type = data_file_folder.split('/')[-1]
            train_file = f'{data_file_folder}/train_data.json'
            validation_file = f'{data_file_folder}/validation_data.json'
            test_file = f'{data_file_folder}/test_data.json'
            for k, learning_rate in enumerate([3e-05]):
                output_dir = os.path.join(
                    models_results_folder,
                    f'start_end_has_answer_use_labels_predictions_{data_type}_{model_name}_original_run_id_{k}')
                if os.path.exists(output_dir):
                    print(f'output_dir {output_dir} exists --> continue')
                    continue
                cache_dir = f'{output_dir}/cache'
                for max_answer_length in [200, 250]:
                    run_model(model_name_or_path, train_file, validation_file, test_file, cache_dir, output_dir,
                              max_answer_length=max_answer_length, use_original_start_end=True,
                              learning_rate=learning_rate,
                              start_end_has_answer_use_predictions_has_answer_in_loss=False)
            for i, model_param_dict in enumerate(model_grid_search_params):
                has_answer_loss_weight = model_param_dict['has_answer_loss_weight']
                start_end_prediction_loss_weight = model_param_dict['start_end_prediction_loss_weight']
                start_end_labels_loss_weight = model_param_dict['start_end_labels_loss_weight']
                learning_rate = model_param_dict['learning_rate']

                output_dir = os.path.join(
                    models_results_folder,
                    f'start_end_has_answer_use_labels_predictions_{data_type}_{model_name}_run_id_{i}')
                if os.path.exists(output_dir):
                    print(f'output_dir {output_dir} exists --> continue')
                    continue
                cache_dir = f'{output_dir}/cache'

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

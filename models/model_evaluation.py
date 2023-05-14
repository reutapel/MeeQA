import collections


def get_tokens_index(s: str, context: str):
    start_position = context.find(s)
    start_index = context[:start_position].count(' ')
    end_index = start_index + s.count(' ')
    return list(range(start_index, end_index + 1))


def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)


def compute_f1(a_gold, a_pred, context):
    gold_toks = get_tokens_index(a_gold, context)
    pred_toks = get_tokens_index(a_pred, context)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def get_raw_scores_subset(dataset, preds):
    exact_scores = {}
    f1_scores = dict()
    precisions = dict()
    recalls = dict()

    for article in dataset:
        qid = article['id']
        gold_answers = [a for a in article['answers']['text']]
        if not gold_answers:
            gold_answers = ['']
        if qid not in preds:
            print('Missing prediction for %s' % qid)
            continue
        a_pred = preds[qid]['text']
        new_f1_sum, new_precision_sum, new_recall_sum = 0.0, 0.0, 0.0
        em_sum = 0.0
        all_subset_best_f1 = 0.0
        num_gold_answers = len(gold_answers)
        if num_gold_answers > 1:
            for i in range(num_gold_answers):
                # exclude the current answer
                current_gold_answers = gold_answers[0:i] + gold_answers[i + 1:]
                em_sum += max(compute_exact(a, a_pred) for a in current_gold_answers)
                new_best_f1, new_best_precision, new_best_recall = 0.0, 0.0, 0.0

                for a in current_gold_answers:  # compare to a subset of n-1 answers
                    new_f1, new_precision, new_recall = compute_f1(a, a_pred, article['context'])
                    new_best_f1 = max(new_best_f1, new_f1)
                    new_best_precision = max(new_best_precision, new_precision)
                    new_best_recall = max(new_best_recall, new_recall)

                    all_subset_best_f1 = max(all_subset_best_f1, new_f1)

                new_f1_sum += new_best_f1
                new_precision_sum += new_best_precision
                new_recall_sum += new_best_recall

        else:  # one gold answer
            a = gold_answers[0]
            em_sum += compute_exact(a, a_pred)
            new_f1, new_precision, new_recall = compute_f1(a, a_pred, article['context'])
            new_f1_sum += new_f1
            new_precision_sum += new_precision
            new_recall_sum += new_recall

        exact_scores[qid] = em_sum / num_gold_answers

        precisions[qid] = new_precision_sum / num_gold_answers
        f1_scores[qid] = new_f1_sum / num_gold_answers
        recalls[qid] = new_recall_sum / num_gold_answers

    return exact_scores, f1_scores, precisions, recalls


def get_raw_scores_human(dataset):
    exact_scores = {}
    f1_scores = dict()
    precisions = dict()
    recalls = dict()
    number_one_annotator = 0

    for article in dataset:
        qid = article['id']
        gold_answers = [a for a in article['answers']['text']]
        if not gold_answers:
            gold_answers = ['']

        new_f1_sum, new_precision_sum, new_recall_sum = 0.0, 0.0, 0.0
        em_sum = 0.0

        num_gold_answers = len(gold_answers)
        if num_gold_answers > 1:
            for i in range(num_gold_answers):
                # exclude the current answer
                current_gold_answers = gold_answers[0:i] + gold_answers[i + 1:]

                new_best_f1, new_best_precision, new_best_recall = 0, 0, 0
                a_pred = gold_answers[i]
                for a in current_gold_answers:
                    new_f1, new_precision, new_recall = compute_f1(a, a_pred, article['context'])
                    new_best_f1 = max(new_best_f1, new_f1)
                    new_best_precision = max(new_best_precision, new_precision)
                    new_best_recall = max(new_best_recall, new_recall)

                new_f1_sum += new_best_f1
                new_precision_sum += new_best_precision
                new_recall_sum += new_best_recall
                em_sum += max(compute_exact(a, gold_answers[i]) for a in current_gold_answers)
        else:
            print(f"One annotator for question {qid}")
            number_one_annotator += 1
            continue

        exact_scores[qid] = em_sum / num_gold_answers

        precisions[qid] = new_precision_sum / num_gold_answers
        f1_scores[qid] = new_f1_sum / num_gold_answers
        recalls[qid] = new_recall_sum / num_gold_answers

    return exact_scores, f1_scores, precisions, recalls


def make_qid_to_answer_type(dataset, answer_type: str):
    qid_to_ans_type = {}
    for article in dataset:
        if answer_type not in article:
            raise ValueError(f'{answer_type} is not in dataset')
        qid_to_ans_type[article['id']] = not article[answer_type]
    return qid_to_ans_type


def make_eval_dict(exact_scores, f1_scores, precision_scores, recall_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('precision', 100.0 * sum(precision_scores.values()) / total),
            ('recall', 100.0 * sum(recall_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list if k in exact_scores) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list if k in f1_scores) / total),
            ('precision', 100.0 * sum(precision_scores[k] for k in qid_list if k in precision_scores) / total),
            ('recall', 100.0 * sum(recall_scores[k] for k in qid_list if k in recall_scores) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def evaluate(eval_dataset, preds=None, data_type: str = 'eval', humans: bool = False):
    if preds is None and not humans:
        raise ValueError('You must give predictions if this is not human evaluation')
    answer_types = [['is_impossible', 'HasAns', 'NoAns']]
    qid_to_ans_type = dict()
    has_ans_type_qids = dict()
    no_ans_type_qids = dict()
    for answer_type, _, _ in answer_types:
        qid_to_ans_type[answer_type] = make_qid_to_answer_type(eval_dataset, answer_type)
        has_ans_type_qids[answer_type] = [k for k, v in qid_to_ans_type[answer_type].items() if v]
        no_ans_type_qids[answer_type] = [k for k, v in qid_to_ans_type[answer_type].items() if not v]

    result = dict()
    if humans:
        exact_raw, f1_raw, precision_raw, recall_raw = get_raw_scores_human(eval_dataset)
    else:
        exact_raw, f1_raw, precision_raw, recall_raw = get_raw_scores_subset(eval_dataset, preds)
    type_result = make_eval_dict(exact_raw, f1_raw, precision_raw, recall_raw)
    merge_eval(result, type_result, f'{data_type}_measures')
    for answer_type, positive_suffix, negative_suffix in answer_types:
        has_ans_qids = has_ans_type_qids[answer_type]
        no_ans_qids = no_ans_type_qids[answer_type]
        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_raw, f1_raw, precision_raw, recall_raw, qid_list=has_ans_qids)
            merge_eval(result, has_ans_eval, f'{data_type}_measures_{positive_suffix}')
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_raw, f1_raw, precision_raw, recall_raw, qid_list=no_ans_qids)
            merge_eval(result, no_ans_eval, f'{data_type}_measures_{negative_suffix}')

    print(f"***** {data_type} results *****")
    for key in sorted(result.keys()):
        print(key, str(round(result[key], 2)))

    return result


def combine_dataset(dataset: list):
    """
    Create a combine dataset where each question
    :param dataset:
    :return:
    """

    new_dataset = collections.defaultdict(list)
    new_dataset_is_impossible = collections.defaultdict(list)
    new_dataset_context = dict()
    final_new_dataset = list()
    for article in dataset:
        q_j_id = article['id']
        q_id = ''.join(q_j_id.split('_')[:-1])
        gold_answers = [a for a in article['answers']['text']]
        gold_answers = ' '.join(gold_answers)
        new_dataset[q_id].append(gold_answers)
        new_dataset_context[q_id] = article['context']
        new_dataset_is_impossible[q_id].append(article['is_impossible'])

    for q_id, answers in new_dataset.items():
        new_item = dict()
        new_item['id'] = q_id
        new_item['answers'] = {'text': answers}
        new_item['context'] = new_dataset_context[q_id]
        new_item['is_impossible'] = majority(new_dataset_is_impossible[q_id])

        final_new_dataset.append(new_item)

    return final_new_dataset


def combine_predictions(predictions):
    """
    Create a combine dataset
    :param predictions:
    :return:
    """

    new_predictions = collections.defaultdict(list)
    final_new_predictions = collections.defaultdict(dict)
    for q_j_id, pred in predictions.items():
        q_id = ''.join(q_j_id.split('_')[:-1])
        new_predictions[q_id].append(pred)

    for q_id, answers in new_predictions.items():
        if len(set(answers)) > 1:
            print('Error: predictions for the same question are not the same')
        final_new_predictions[q_id]['text'] = answers[0]

    return final_new_predictions


def majority(arr):
    # convert array into dictionary
    freqDict = collections.Counter(arr)

    # traverse dictionary and check majority element
    size = len(arr)
    for (key, val) in freqDict.items():
        if val > (size / 2):
            return key
    return False


def model_results_evaluation(dataset, predictions, data_type: str = 'eval'):
    new_predictions = combine_predictions(predictions)
    new_test_data = combine_dataset(dataset)
    result_prediction_statistics_dict = evaluate(new_test_data, new_predictions, data_type=data_type)

    return result_prediction_statistics_dict

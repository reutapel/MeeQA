import copy
import json
import pandas as pd
import os
import re


parent = os.path.join(os.getcwd(), os.pardir)
data_folder = os.path.join(parent, 'Data')
if not os.path.exists(data_folder):
    os.mkdir(data_folder)
number_of_utterances_before = 1


def huggingface_format_no_answer(paragraph_dict, row, question, answer_text: list, start_position: list):
    paragraph_dict['answers'] = {'text': answer_text, 'answer_start': start_position}
    paragraph_dict['question'] = question
    paragraph_dict['id'] = row.question_id + '_' + str(int(row.JudgeID))
    paragraph_dict['is_impossible'] = row.hasNoAnswer
    paragraph_dict['isQuestionComprehensible'] = row.isQuestionComprehensible
    paragraph_dict['isQuestionRelavent'] = row.isQuestionRelavent
    paragraph_dict['isQuestionSelfContained'] = row.isQuestionSelfContained
    paragraph_dict['isQuestionYesNo'] = row.isQuestionYesNo
    return paragraph_dict


def create_pre_question_context(pre_sentences):
    pre_sentences = pre_sentences.split('& ')
    pre_sentences.remove('')
    pre_sentences = [f'& {text}' for text in pre_sentences[-number_of_utterances_before:]]
    pre_sentences = ''.join(pre_sentences)
    pre_sentences = remove_comments_from_context(pre_sentences)
    return pre_sentences


def remove_comments_from_context(context):
    start_with_umper = True if context[0] == '&' else False
    context_list = context.split('& ')
    for i, utterance in enumerate(context_list):
        if 'statement -- see appendix' in utterance or "secretary 's note" in utterance:
            # find note
            if 'statement -- see appendix' in utterance:
                start_note = utterance.find('statement -- see appendix')
            else:
                start_note = utterance.find("secretary 's note")
            end_note = start_note + utterance[start_note:].find('.')
            new_utterance = utterance[:start_note] + utterance[end_note+1:]
            new_utterance = new_utterance.replace('  ', ' ')
            pattern1 = r'SPEAKER_([0-9]): '
            pattern2 = r'SPEAKER_([0-9])([0-9]): '
            pattern3 = r'SPEAKER_([0-9]):'
            pattern4 = r'SPEAKER_([0-9])([0-9]):'
            pattern_match1 = re.match(pattern1, new_utterance)
            pattern_match2 = re.match(pattern2, new_utterance)
            pattern_match3 = re.match(pattern3, new_utterance)
            pattern_match4 = re.match(pattern4, new_utterance)
            pattern_match1 = pattern_match1[0] if pattern_match1 is not None else ''
            pattern_match2 = pattern_match2[0] if pattern_match2 is not None else ''
            pattern_match3 = pattern_match3[0] if pattern_match3 is not None else ''
            pattern_match4 = pattern_match4[0] if pattern_match4 is not None else ''

            if new_utterance == ' .' or new_utterance == '.' or pattern_match1 == new_utterance or\
                    pattern_match2 == new_utterance or pattern_match3 == new_utterance or\
                    pattern_match4 == new_utterance:
                context_list[i] = ''
            else:
                context_list[i] = new_utterance
    context_list = [item for item in context_list if item != '']
    context = '& '.join(context_list)
    if start_with_umper:
        return f'& {context}'
    else:
        return context


def create_only_switch_speakers(context: str, question_text_speaker: str):
    context_list = context.split('& ')
    if context_list[0] == '':
        context_list = context_list[1:]
        start_with_umper = True
    else:
        start_with_umper = False
    # start with the question speaker
    prev_speaker = question_text_speaker[question_text_speaker.find('SPEAKER'): question_text_speaker.find(':') + 1]
    for i, utterance in enumerate(context_list):
        if utterance == '':
            continue
        speaker = utterance[utterance.find('SPEAKER'): utterance.find(':') + 1]
        if speaker == prev_speaker:  # remove the speaker
            context_list[i] = utterance[utterance.find(':') + 2:]
        else:
            if not start_with_umper and i == 0:
                context_list[i] = f'{utterance}'
            else:
                context_list[i] = f'& {utterance}'
        prev_speaker = speaker if speaker != '' else prev_speaker  # for answers in the middle of the utterance

    indices_to_remove = [i for i, ltr in enumerate(context_list) if ltr == '']
    indices_to_remove = sorted(indices_to_remove, reverse=True)

    for ind in indices_to_remove:
        if ind < len(context_list):
            context_list.pop(ind)

    context = ''.join(context_list)
    for punc in ['.', ',', '?', ':', '!']:
        context = context.replace(f' {punc}', punc)
    if context[-1] == ' ' or context[-1] == '&':
        context = context[:-1]
    if context[0] == ' ':
        context = context[1:]

    context = context.replace('  ', ' ')
    return context


def create_1_pre_utterance_only_switch(pre_question: str, question: str):
    pre_speaker = pre_question[pre_question.find('_')+1: pre_question.find(':')]
    if pre_speaker == '0':  # same speaker as the question
        question = question.replace('& SPEAKER_0: ', '')
    final_question = pre_question + question

    return final_question


def create_qa_has_answer_model_input_for_meeting(meeting_data: pd.DataFrame, pre_q_sen: bool = False,
                                                 only_switch_speaker: bool = False):
    """
    Create model for a specific meeting. The format is the same as of SQuAD.
    Context is all the post_question_sentences.
    If there is more than one span --> create answer from the beginning of the first span to end of the last span
    :param meeting_data:
    :param pre_q_sen:
    :param only_switch_speaker: keep the speaker only if there was a switch
    :return:
    """

    meeting_data_list = list()
    meeting_data = meeting_data.loc[meeting_data.is_question_check]
    for index, row in meeting_data.iterrows():
        question_dict = dict()
        question_dict['title'] = ''
        if type(row.post_question_sentences_speakers) == float:
            print('type(row.post_question_sentences_speakers) == float')
            continue
        pre_sentences = row.pre_question_sentences_speakers
        if type(pre_sentences) == str:
            pre_sentences = create_pre_question_context(pre_sentences)
        else:
            pre_sentences = ''
        if number_of_utterances_before == 1 and only_switch_speaker and pre_q_sen:
            question = create_1_pre_utterance_only_switch(pre_sentences, row.question_text_speaker)
        else:
            question = row.question_text_speaker if not pre_q_sen else \
                pre_sentences + row.question_text_speaker

        context = row.post_question_sentences_speakers

        original_context = copy.deepcopy(context)
        original_context = original_context.replace('  ', ' ')
        if only_switch_speaker:
            context = create_only_switch_speakers(context, row.question_text_speaker)

        context = remove_comments_from_context(context)
        paragraph_dict = {'context': context}

        # for multiple span: create multiple answers
        if row.hasNoAnswer or len(eval(row.answer_position_with_speakers)) == 0:
            row.hasNoAnswer = True
            answer_text = ['']
            start_position = [-1]
            question_dict.update(huggingface_format_no_answer(paragraph_dict, row, question, answer_text,
                                                              start_position))
            meeting_data_list.append(question_dict)
        else:
            answer_positions = eval(row.answer_position_with_speakers)
            row.hasNoAnswer = False
            start_position = list()
            answer_text_list = list()
            for i, position in enumerate(answer_positions):
                answer_text = eval(row.answer_text_with_speakers)[i]
                answer_text = remove_comments_from_context(answer_text)
                if answer_text == '& ' or answer_text == '& &':
                    print(f'answer_text is: {answer_text}, question ID is: {row.question_id}')
                    continue
                if only_switch_speaker:
                    # add the first speaker, answer starts at start of the utterance
                    context_to_find_speaker = original_context[position[0] - 13:position[0]]
                    if ' SPEAKER' in context_to_find_speaker:
                        first_speaker_answer = context_to_find_speaker[context_to_find_speaker.find('_') + 1:
                                                                       context_to_find_speaker.find(':')]
                        answer_text = f'& SPEAKER_{first_speaker_answer}: ' + answer_text
                    answer_new_position = original_context.find(answer_text)
                    # answer starts at start of first utterance
                    if answer_new_position == 0:
                        prev_speaker_text = row.question_text_speaker
                    else:  # find prev speaker
                        j = 1
                        while original_context[answer_new_position - j] != '&' and answer_new_position - j >= 0:
                            j += 1
                        prev_speaker_text = original_context[answer_new_position - j:]

                    answer_text = create_only_switch_speakers(answer_text, prev_speaker_text)
                answer_text_list.append(answer_text)
                start_position.append(context.find(answer_text))
                if context.find(answer_text) == -1:
                    print(f'start_position is -1, answer_text: {answer_text}, original_context: {original_context}')
                    continue
            if len(answer_text_list) == 0:
                continue
            question_dict.update(huggingface_format_no_answer(paragraph_dict, row, question, answer_text_list,
                                                              start_position))

            meeting_data_list.append(question_dict)

    return meeting_data_list


def main(pre_q_sen, only_switch_speaker):
    inner_dir = ''
    if pre_q_sen:
        inner_dir = f'{number_of_utterances_before}_pre_'
    if only_switch_speaker:
        inner_dir += 'only_switch_speaker'
    if inner_dir == '':
        inner_dir = 'original'
    all_data_dir = os.path.join(data_folder, inner_dir)
    if not os.path.exists(all_data_dir):
        os.mkdir(all_data_dir)

    all_data = {'train': list(), 'validation': list(), 'test': list()}
    train_meetings = list()
    validation_meetings = list()
    test_meetings = list()

    all_original_data = pd.read_csv(os.path.join(data_folder, 'all_original_data.csv'))

    train_meetings_list = pd.read_csv(os.path.join(data_folder, 'train_meetings.csv')).meeting_name.to_list()
    validation_meetings_list = pd.read_csv(os.path.join(data_folder, 'validation_meetings.csv')).meeting_name.to_list()
    test_meetings_list = pd.read_csv(os.path.join(data_folder, 'test_meetings.csv')).meeting_name.to_list()

    for i, meeting in enumerate(train_meetings_list):
        print(f'Working on train meeting number {i}: {meeting}')
        meeting_data = all_original_data.loc[all_original_data.raw_Transcripte_Name == meeting]
        meeting_test_data = create_qa_has_answer_model_input_for_meeting(
            meeting_data, pre_q_sen=pre_q_sen, only_switch_speaker=only_switch_speaker)
        train_meetings.append(meeting)
        all_data['train'] += meeting_test_data
    for i, meeting in enumerate(validation_meetings_list):
        print(f'Working on validation meeting number {i}: {meeting}')
        meeting_data = all_original_data.loc[all_original_data.raw_Transcripte_Name == meeting]
        meeting_test_data = create_qa_has_answer_model_input_for_meeting(
            meeting_data, pre_q_sen=pre_q_sen, only_switch_speaker=only_switch_speaker)
        validation_meetings.append(meeting)
        all_data['validation'] += meeting_test_data
    for i, meeting in enumerate(test_meetings_list):
        print(f'Working on test meeting number {i}: {meeting}')
        meeting_data = all_original_data.loc[all_original_data.raw_Transcripte_Name == meeting]
        meeting_test_data = create_qa_has_answer_model_input_for_meeting(
            meeting_data, pre_q_sen=pre_q_sen, only_switch_speaker=only_switch_speaker)
        all_data['test'] += meeting_test_data
        test_meetings.append(meeting)

    # final data for model after split
    print(f'Save data')
    for data_type, data in all_data.items():
        all_data_dict = {'data': data}
        with open(os.path.join(all_data_dir, f'{data_type}_data.json'), 'w') as outfile:
            json.dump(all_data_dict, outfile)


if __name__ == '__main__':
    # main(pre_q_sen=False, only_switch_speaker=True)
    # main(pre_q_sen=False, only_switch_speaker=False)
    main(pre_q_sen=True, only_switch_speaker=True)

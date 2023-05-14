""" DebertaForQuestionAnsweringHasAnswerUsePredictions model. """

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.utils import logging
from transformers.models.deberta.modeling_deberta import DebertaModel, DebertaPreTrainedModel
from QuestionAnsweringHasAnswerModelOutput import QuestionAnsweringHasAnswerOutputModelOutput


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaConfig"
_TOKENIZER_FOR_DOC = "DebertaTokenizer"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-base"


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention
    <https://arxiv.org/abs/2006.03654>`_ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build on top of
    BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.```


    Parameters:
        config (:class:`~transformers.DebertaConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.DebertaTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering (a linear layers on top of 
    the hidden-states output to compute `span start logits` and `span end logits`), and a HasAnswer classification
    head on top. The loss function is the flat-hierarchical loss (FHL) as describe in our paper.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForQuestionAnsweringHasAnswerUsePredictions(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, has_answer_loss_weight: float = 1.0, start_end_prediction_loss_weight: float = 1.0,
                 start_end_labels_loss_weight: float = 1.0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.has_answer_loss_weight = has_answer_loss_weight
        self.start_end_prediction_loss_weight = start_end_prediction_loss_weight
        self.start_end_labels_loss_weight = start_end_labels_loss_weight

        self.softmax = torch.nn.Softmax(dim=1)

        self.deberta = DebertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_answer = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
            tokenizer_class=_TOKENIZER_FOR_DOC,
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=QuestionAnsweringModelOutput,
            config_class=_CONFIG_FOR_DOC,
        )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_impossible=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        is_impossible
            Labels for each sample if there is an answer or not. 1 if there is no answer
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        no_answer_logits = self.has_answer(cls_output)

        no_answer_prob = self.softmax(no_answer_logits)
        # use the probability for no answer
        no_answer_prob = no_answer_prob[:, 1]

        total_loss = None
        if start_positions is not None and end_positions is not None and is_impossible is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            is_impossible = is_impossible.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            has_answer_loss_fct = CrossEntropyLoss()

            all_total_losses = list()
            for i in range(is_impossible.shape[0]):
                start_loss = loss_fct(start_logits[i].view(-1, start_logits.shape[1]), start_positions[i].view(-1))
                end_loss = loss_fct(end_logits[i].view(-1, end_logits.shape[1]), end_positions[i].view(-1))
                start_end_loss = (start_loss + end_loss) / 2
                has_answer_loss =\
                    has_answer_loss_fct(no_answer_logits[i].view(-1, self.num_labels), is_impossible[i].view(-1))
                sample_total_loss = self.has_answer_loss_weight * has_answer_loss + \
                                    self.start_end_prediction_loss_weight * (1 - no_answer_prob[i]) * start_end_loss + \
                                    self.start_end_labels_loss_weight * (1 - is_impossible[i]) * start_end_loss
                all_total_losses.append(sample_total_loss)
            total_loss = sum(all_total_losses) / len(all_total_losses)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringHasAnswerOutputModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            sequence_logits=no_answer_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

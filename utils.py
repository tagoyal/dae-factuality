from torch import nn
import torch, os, logging, csv, copy, math
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)
import sys
import utils
sys.modules['train_importance_utils'] = utils

class BERTBasicModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTBasicModel, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)

        self.init_weights()

    def forward(self, input_ids, attention, token_ids, child, head, context, dep_labels, num_dependency, arcs,
                arc_label_lengths, arc_ids, sent_label, device='cuda'):
        transformer_outputs = self.bert(input_ids, attention_mask=attention, token_type_ids=token_ids)
        output = transformer_outputs[0]

        output = self.dropout(output)
        output_pooled = output[:, 0]

        logits_all = self.classifier(output_pooled)

        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits_all.view(-1, 3), sent_label.view(-1))

        outputs_return = (logits_all, None, None, None)
        outputs_return = (loss,) + outputs_return

        return outputs_return


class BertDAEModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertDAEModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dep_label_classifier = nn.Linear(3 * config.hidden_size, 2)

        self.sigmoid = nn.Sigmoid()
        self.importance_regression = nn.Linear(3 * config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def forward(self, input_ids, attention, token_ids, child, head, context, dep_labels, num_dependency, arcs,
                arc_label_lengths, arc_ids, sent_label, device='cuda'):
        batch_size = input_ids.size(0)

        transformer_outputs = self.bert(input_ids, attention_mask=attention, token_type_ids=token_ids)
        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        outputs = outputs.view((-1, outputs.size(-1)))

        add = torch.arange(batch_size) * input_ids.size(1)
        add = add.unsqueeze(1).to(device)
        child_temp = child + add
        head_temp = head + add

        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]

        child_embeddings = child_embeddings.view(batch_size, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))

        arcs = arcs.view(-1, arcs.size(-1))
        arc_label_lengths = arc_label_lengths.view(-1)
        arc_attention = torch.arange(arcs.size(1)).to(device)[None, :] <= arc_label_lengths[:, None]
        arc_attention = arc_attention.type(torch.float)
        arc_outputs = self.bert(arcs, attention_mask=arc_attention)
        arc_outputs = arc_outputs[1]
        arc_outputs = self.dropout(arc_outputs)
        arc_outputs = arc_outputs.view(batch_size, -1, arc_outputs.size(1))

        final_embeddings = torch.cat([child_embeddings, head_embeddings, arc_outputs], dim=2)
        logits_all = self.dep_label_classifier(final_embeddings)

        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits_all.view(-1, 2), dep_labels.view(-1))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return


class ElectraBasicModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)

        self.init_weights()

    def forward(self, input_ids, attention, token_ids, child, head, context, dep_labels, num_dependency, arcs,
                arc_label_lengths, arc_ids, sent_label, device='cuda'):
        transformer_outputs = self.electra(input_ids, attention_mask=attention, token_type_ids=token_ids)
        output = transformer_outputs[0]

        output = self.dropout(output)
        output_pooled = output[:, 0]

        logits_all = self.classifier(output_pooled)

        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits_all.view(-1, 3), sent_label.view(-1))

        outputs_return = (logits_all, None, None, None)
        outputs_return = (loss,) + outputs_return

        return outputs_return


class ElectraDAEModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dep_label_classifier = nn.Linear(3 * config.hidden_size, 2)

        self.sigmoid = nn.Sigmoid()
        self.importance_regression = nn.Linear(3 * config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def forward(self, input_ids, attention, token_ids, child, head, dep_labels, arcs, arc_label_lengths, device='cuda'):
        batch_size = input_ids.size(0)

        transformer_outputs = self.electra(input_ids, attention_mask=attention, token_type_ids=token_ids)
        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        outputs = outputs.view((-1, outputs.size(-1)))

        add = torch.arange(batch_size) * input_ids.size(1)
        add = add.unsqueeze(1).to(device)
        child_temp = child + add
        head_temp = head + add

        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]

        child_embeddings = child_embeddings.view(batch_size, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))

        arcs = arcs.view(-1, arcs.size(-1))
        arc_label_lengths = arc_label_lengths.view(-1)
        arc_attention = torch.arange(arcs.size(1)).to(device)[None, :] <= arc_label_lengths[:, None]
        arc_attention = arc_attention.type(torch.float)
        transformer_arc_outputs = self.electra(arcs, attention_mask=arc_attention)
        arc_outputs = transformer_arc_outputs[0]
        pooled_arc_outputs = arc_outputs[:, 0]
        pooled_arc_outputs = self.dropout(pooled_arc_outputs)
        pooled_arc_outputs = pooled_arc_outputs.view(batch_size, -1, pooled_arc_outputs.size(1))

        final_embeddings = torch.cat([child_embeddings, head_embeddings, pooled_arc_outputs], dim=2)
        logits_all = self.dep_label_classifier(final_embeddings)

        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits_all.view(-1, 2), dep_labels.view(-1))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return


def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def get_train_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "train.tsv"))


def get_dev_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "dev.tsv"))


def pad_1d(input, max_length, pad_token):
    padding_length = max_length - len(input)
    if padding_length < 0:
        input = input[:max_length]
        padding_length = 0
    input = input + ([pad_token] * padding_length)
    return input


class InputFeatures(object):
    def __init__(self, input_ids, input_attention_mask, sentence_label, child_indices, head_indices,
                 dep_labels, num_dependencies, arcs, arc_labels, arc_label_lengths, token_ids=None):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.token_ids = token_ids
        self.sentence_label = sentence_label
        self.child_indices = child_indices
        self.head_indices = head_indices
        self.dep_labels = dep_labels
        self.num_dependencies = num_dependencies
        self.arcs = arcs
        self.arc_labels = arc_labels
        self.arc_label_lengths = arc_label_lengths


def convert_examples_to_features_bert(examples, tokenizer, max_length=128, pad_token=None, pad_token_segment_id=None,
                                      num_deps_per_ex=20):
    features = []
    rejected_ex = 0

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        tokens_input = []
        tokens_input.extend(tokenizer.tokenize('[CLS]'))
        index_now = len(tokens_input)
        for (word_index, word) in enumerate(example['input'].split(' ')):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_input.extend(word_tokens)
                index_now += len(word_tokens)

        tokens_input.extend(tokenizer.tokenize('[SEP]'))
        index_now += 1

        index_map = {}
        for (word_index, word) in enumerate(example['context'].split(' ')):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_input.extend(word_tokens)
                index_now += len(word_tokens)
                index_map[word_index] = index_now - 1
        tokens_input.extend(tokenizer.tokenize('[SEP]'))

        child_indices = [0] * num_deps_per_ex
        head_indices = [0] * num_deps_per_ex
        dep_labels = [-1] * num_deps_per_ex
        num_dependencies = 0

        input_arcs = [[0] * 20 for _ in range(num_deps_per_ex)]
        arc_labels = [[0] * 10 for _ in range(num_deps_per_ex)]
        arc_label_lengths = [0] * num_deps_per_ex

        for i in range(num_deps_per_ex):
            if example['dep_idx' + str(i)] == '':
                break

            child_idx, head_idx = example['dep_idx' + str(i)].split(' ')
            child_idx = int(child_idx)
            head_idx = int(head_idx)

            num_dependencies += 1
            dep_labels[i] = int(example['dep_label' + str(i)])
            child_indices[i] = index_map[child_idx]
            head_indices[i] = index_map[head_idx]
            arc_label_ids = tokenizer.encode(example['dep' + str(i)])
            arc_label_lengths[i] = len(arc_label_ids) - 1
            arc_labels[i] = pad_1d(arc_label_ids, 10, pad_token)

            w1 = example['dep_words' + str(i)].split(' ')[0]
            w2 = example['dep_words' + str(i)].split(' ')[1]
            arc_text = example['dep' + str(i)] + ' [SEP] ' + w1 + ' [SEP] ' + w2
            arc = tokenizer.encode(arc_text)
            input_arcs[i] = pad_1d(arc, 20, pad_token)

        if len(tokens_input) > max_length:
            rejected_ex += 1
            # tokens_input = tokens_input[:max_length]
            continue

        if num_dependencies == 0:
            rejected_ex += 1
            continue

        sentence_label = int(example['sentlabel'])

        inputs = tokenizer.encode_plus(example['input'], example['context'], add_special_tokens=True)
        input_ids = inputs['input_ids']
        token_ids = inputs['token_type_ids']

        assert len(tokens_input) == len(input_ids), 'length mismatched'
        padding_length_a = max_length - len(tokens_input)
        input_ids = input_ids + ([pad_token] * padding_length_a)
        token_ids = token_ids + ([0] * padding_length_a)
        input_attention_mask = [1] * len(tokens_input) + ([0] * padding_length_a)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_attention_mask=input_attention_mask,
                                      token_ids=token_ids,
                                      sentence_label=sentence_label,
                                      child_indices=child_indices,
                                      head_indices=head_indices,
                                      dep_labels=dep_labels,
                                      num_dependencies=num_dependencies,
                                      arcs=input_arcs,
                                      arc_labels=arc_labels,
                                      arc_label_lengths=arc_label_lengths))
    return features


def load_and_cache_examples_bert(args, tokenizer, evaluate):
    if evaluate:
        data_dir = '/'.join(args.eval_data_file.split('/')[:-1])
    else:
        data_dir = '/'.join(args.train_data_file.split('/')[:-1])

    model_type = args.model_type

    if 'bert' in args.model_type:  # hack
        model_type = 'bert'

    if 'electra' in args.model_type:
        model_type = 'electra'

    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            model_type,
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = (get_dev_examples(data_dir) if evaluate else get_train_examples(data_dir))
        features = convert_examples_to_features_bert(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_attention_mask = torch.tensor([f.input_attention_mask for f in features], dtype=torch.long)
    input_token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)

    child_indices = torch.tensor([f.child_indices for f in features], dtype=torch.long)
    head_indices = torch.tensor([f.head_indices for f in features], dtype=torch.long)

    dep_labels = torch.tensor([f.dep_labels for f in features], dtype=torch.long)
    num_dependencies = torch.tensor([f.num_dependencies for f in features], dtype=torch.long)
    arcs = torch.tensor([f.arcs for f in features], dtype=torch.long)
    arc_labels = torch.tensor([f.arc_labels for f in features], dtype=torch.long)
    arc_label_lengths = torch.tensor([f.arc_label_lengths for f in features], dtype=torch.long)

    sentence_label = torch.tensor([f.sentence_label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, input_attention_mask, input_token_ids, child_indices, head_indices,
                            dep_labels, num_dependencies, arcs, arc_labels, arc_label_lengths, sentence_label)

    return dataset

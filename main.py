import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import csv
from sklearn.utils.extmath import softmax
from transformers import glue_compute_metrics as compute_metrics
import utils

from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {"bert_dae": (BertConfig, utils.BertDAEModel, BertTokenizer),
                 "bert_basic": (BertConfig, utils.BERTBasicModel, BertTokenizer),
                 "electra_basic": (ElectraConfig, utils.ElectraBasicModel, ElectraTokenizer),
                 "electra_dae": (ElectraConfig, utils.ElectraDAEModel, ElectraTokenizer), }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def save_checkpoints(args, output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def compute_metrics_intermediate(preds, gold):
    preds_new = []
    gold_new = []
    for p, g in zip(preds, gold):
        if g == -1:
            continue
        else:
            preds_new.append(p)
            gold_new.append(g)

    preds_new = np.array(preds_new)
    gold_new = np.array(gold_new)

    result = compute_metrics('qqp', preds_new, gold_new)
    return result


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = utils.load_and_cache_examples_bert(args, tokenizer, evaluate)
    return dataset


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            input_ids, attention, token_ids, child, head = batch[0], batch[1], batch[2], batch[3], batch[4]
            dep_labels, num_dependency, arcs, arc_labels = batch[5], batch[6], batch[7], batch[8]
            arc_label_lengths, sent_labels = batch[9], batch[10]

            inputs = {'input_ids': input_ids, 'attention': attention, 'token_ids': token_ids, 'child': child,
                      'head': head, 'dep_labels': dep_labels, 'arcs': arc_labels,
                      'arc_label_lengths': arc_label_lengths, 'device': args.device}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = dep_labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, dep_labels.detach().cpu().numpy(), axis=0)

    f_out = open(os.path.join(eval_output_dir, 'dev_out.txt'), 'w')
    k = 0
    for batch in eval_dataloader:
        for inp, arc_list in zip(batch[0], batch[8]):
            text = tokenizer.decode(inp)
            text = text.replace(tokenizer.pad_token, '').strip()
            f_out.write(text + '\n')

            for j, arc in enumerate(arc_list):
                arc_text = tokenizer.decode(arc)
                arc_text = arc_text.replace(tokenizer.pad_token, '').strip()

                if arc_text == '':  # for bert
                    break

                pred_temp = softmax([preds[k][j]])

                f_out.write(text + '\n')
                f_out.write(arc_text + '\n')
                f_out.write('gold:\t' + str(out_label_ids[k][j]) + '\n')
                f_out.write('pred:\t' + str(np.argmax(pred_temp)) + '\n')
                f_out.write(str(pred_temp[0][0]) + '\t' + str(pred_temp[0][1]) + '\n')
                f_out.write('\n')

            k += 1

    f_out.close()

    preds = preds.reshape(-1, 2)
    preds = softmax(preds)
    out_label_ids = out_label_ids.reshape(-1)
    preds = np.argmax(preds, axis=1)

    result = compute_metrics_intermediate(preds, out_label_ids)
    print(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("dep level %s = %s", key, str(result[key]))
            writer.write("dep level  %s = %s\n" % (key, str(result[key])))
        writer.write('\n')

    return result


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, tr_loss_sent, logging_loss, logging_loss_sent = 0.0, 0.0, 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproducibility

    results = {}
    acc_prev = 0.
    preds = None
    labels = None

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = tuple(t.to(args.device) for t in batch)
            input_ids, attention, token_ids, child, head = batch[0], batch[1], batch[2], batch[3], batch[4]
            dep_labels, num_dependency, arcs, arc_labels = batch[5], batch[6], batch[7], batch[8]
            arc_label_lengths, sent_labels = batch[9], batch[10]

            inputs = {'input_ids': input_ids, 'attention': attention, 'token_ids': token_ids, 'child': child,
                      'head': head, 'dep_labels': dep_labels, 'arcs': arc_labels,
                      'arc_label_lengths': arc_label_lengths, 'device': args.device}

            model.train()
            outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]

            tr_loss += loss.item()

            loss.backward()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = dep_labels.view(-1).cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, dep_labels.view(-1).cpu().numpy(), axis=0)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    logs = {}
                    loss_scalar_dep = (tr_loss - logging_loss) / args.save_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss_dep"] = loss_scalar_dep
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    preds = preds.reshape(-1, 2)
                    preds = softmax(preds)
                    preds = np.argmax(preds, axis=1)
                    res_train = compute_metrics_intermediate(preds, labels)
                    preds = None
                    labels = None

                    print(res_train)
                    # Evaluation
                    result = evaluate(args, model, tokenizer)
                    results.update(result)

                    save_checkpoints(args, args.output_dir, model, tokenizer)

                    if result['acc'] > acc_prev:
                        acc_prev = result['acc']
                        # Save model checkpoint best
                        output_dir = os.path.join(args.output_dir, "model-best")
                        save_checkpoints(args, output_dir, model, tokenizer)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        required=True,
        help="Evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file)."
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size evaluation.", )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--seed", type=int, default=43, help="random seed for initialization")

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda", args.gpu_device)
    #device = torch.device("cuda")
    print(device)
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, 'model.log')
    )

    # Set seed
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.input_dir is not None:
        print('loading model')
        tokenizer = tokenizer_class.from_pretrained(args.input_dir)
        model = model_class.from_pretrained(args.input_dir)
    else:
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    evaluate(args, model, tokenizer)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()

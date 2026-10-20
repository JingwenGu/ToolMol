"""Standalone SFT trainer for the online-SFT analogue of online few-shot (see online_sft.py).

Run as a subprocess (not imported into the main GA process) so that torch/transformers/
bitsandbytes are only ever loaded when --online_fewshot_mode sft is actually used, and so a
training crash surfaces as a nonzero exit code rather than taking down the GA run.

Takes a JSON file of buffer entries (the same shape OnlineFewShotBuffer._buffer_entries
holds - each a dict with a 'messages' conversation, see online_fewshot.py's _build_messages)
and fine-tunes --base_model on them from scratch (never continuing a previous SFT round - see
online_sft.py's module docstring for why), saving the result to --output_dir.

Each buffer entry's 'messages' is one continuous multi-turn conversation (not split into one
example per edit step - see the note in online_fewshot.py's _build_messages and the PR
discussion this followed): a full user/assistant/tool/.../assistant("FINAL ANSWER")
exchange. Training keeps that structure - one example per entry, a system prompt prepended -
and masks the loss to assistant-turn tokens only, found by the standard incremental-render
technique (re-render the conversation prefix before and through each assistant turn with the
real chat template + tool schema, and take the token-length difference as that turn's span).
This matters more than it would for plain chat SFT, since the chat template renders a
<tool_call>{...}</tool_call> block as part of the assistant turn's content - that's the span
we need the model to learn to produce, not just the free-text reasoning around it.

Usage:
    python main/toolmol/online_sft_train.py \\
        --base_model Qwen/Qwen3-4B-Instruct-2507 \\
        --train_file /path/to/entries.json \\
        --output_dir /path/to/checkpoint_round_N \\
        --epochs 10
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

from main.toolmol.agent import SYSTEM_PROMPT
from main.toolmol.toolbox import TOOL_SCHEMAS


def _build_conversation(entry):
    return [{"role": "system", "content": SYSTEM_PROMPT}] + entry["messages"]


def _render(tokenizer, conversation, upto, add_generation_prompt):
    # upto: number of leading messages to include (a prefix of `conversation`).
    return tokenizer.apply_chat_template(
        conversation[:upto], tools=TOOL_SCHEMAS, tokenize=True,
        add_generation_prompt=add_generation_prompt)


def _build_example(tokenizer, entry):
    """Returns (input_ids, labels) for one buffer entry, with labels == -100 everywhere
    except the token span of each assistant turn (including its rendered tool_calls),
    which is set to the matching input_ids so the model is trained to produce exactly
    that span - the standard ignore_index=-100 convention transformers' default
    cross-entropy loss already honors."""
    conversation = _build_conversation(entry)
    full_ids = _render(tokenizer, conversation, len(conversation), add_generation_prompt=False)
    labels = [-100] * len(full_ids)

    for i, msg in enumerate(conversation):
        if msg["role"] != "assistant":
            continue
        # The prefix up to (not including) this assistant turn, rendered as if about to
        # generate it - this is exactly the text this turn's tokens get appended after.
        prefix_ids = _render(tokenizer, conversation, i, add_generation_prompt=True)
        # The same prefix plus this turn, with no further generation prompt appended.
        through_ids = _render(tokenizer, conversation, i + 1, add_generation_prompt=False)
        start, end = len(prefix_ids), len(through_ids)
        if through_ids[:start] != prefix_ids:
            raise ValueError(
                f"chat template did not render turn {i} as a clean append - prefix/through "
                f"diverge before the turn even starts, so this turn's span can't be located")
        if end > len(full_ids):
            raise ValueError(
                f"turn {i}'s rendered-through length ({end}) exceeds the full conversation's "
                f"({len(full_ids)}) - the template may not be append-only across later turns")
        labels[start:end] = full_ids[start:end]

    return full_ids, labels


def _collate(batch, pad_token_id):
    import torch
    max_len = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention_mask = [], [], []
    for ids, lab in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_file", required=True, help="JSON list of buffer entries")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    with open(args.train_file, encoding="utf-8") as f:
        entries = json.load(f)
    if not entries:
        print("online_sft_train: no entries to train on, exiting without writing a checkpoint", flush=True)
        sys.exit(1)

    print(f"online_sft_train: loading {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    print(f"online_sft_train: building {len(entries)} training example(s)...", flush=True)
    examples = [_build_example(tokenizer, e) for e in entries]
    for ids, lab in examples:
        n_trainable = sum(1 for x in lab if x != -100)
        print(f"  example: {len(ids)} tokens, {n_trainable} trainable (assistant-turn) tokens", flush=True)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, idx):
            return examples[idx]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_Dataset(),
        data_collator=lambda batch: _collate(batch, pad_token_id),
    )
    trainer.train()

    print(f"online_sft_train: saving fine-tuned checkpoint to {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("online_sft_train: done", flush=True)


if __name__ == "__main__":
    main()

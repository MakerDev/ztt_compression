"""Dataset preparation utilities."""

from typing import Any, Dict

from datasets import load_dataset


def _format_mmlu_prompt(question: str, choices: list[str], answer: int | None = None) -> str:
    prompt = f"Question: {question}\n\n"
    for index, choice in enumerate(choices):
        prompt += f"{chr(65 + index)}. {choice}\n"
    if answer is not None:
        prompt += f"\nAnswer: {chr(65 + answer)}"
    else:
        prompt += "\nAnswer:"
    return prompt


def _format_mmlu_dataset(examples: Dict[str, Any]) -> Dict[str, Any]:
    texts: list[str] = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        if "choices" in examples:
            choices = examples["choices"][i]
        else:
            choices = [
                examples.get("choice_a", examples.get("A", [""]))[i],
                examples.get("choice_b", examples.get("B", [""]))[i],
                examples.get("choice_c", examples.get("C", [""]))[i],
                examples.get("choice_d", examples.get("D", [""]))[i],
            ]
        answer = examples["answer"][i]
        if isinstance(answer, str):
            answer = ord(answer.upper()) - 65
        texts.append(_format_mmlu_prompt(question, choices, answer))
    return {"text": texts}


def prepare_dataset(
    name: str,
    tokenizer: Any,
    max_seq_length: int,
    subset_ratio: float | None = None,
) -> Any:
    """Load and prepare datasets for training."""
    if name == "mmlu_aux":
        dataset = load_dataset("kz919/mmlu-auxiliary-train-auto-labelled", split="train")
        if subset_ratio:
            dataset = dataset.select(range(int(len(dataset) * subset_ratio)))
        dataset = dataset.map(_format_mmlu_dataset, batched=True)
        return dataset

    if name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test")
        if subset_ratio:
            dataset = dataset.select(range(int(len(dataset) * subset_ratio)))
        dataset = dataset.map(_format_mmlu_dataset, batched=True)
        return dataset

    raise ValueError(f"Unknown dataset name: {name}")

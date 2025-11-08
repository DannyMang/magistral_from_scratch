"""
Download and prepare datasets for Magistral-style RL training.
Focuses on verifiable math and code problems.
"""

from datasets import load_dataset, DatasetDict
import json
from pathlib import Path
from tqdm import tqdm

def download_math_data() -> tuple[DatasetDict, DatasetDict]:
    gsm8k: DatasetDict = load_dataset("gsm8k", "main")  # type: ignore
    print("loaded GSM8K...")
    math: DatasetDict = load_dataset("hendrycks/competition_math")  # type: ignore
    print("loading hendrycks MATH...")

    # Save raw data
    Path("./raw/math").mkdir(parents=True, exist_ok=True)
    gsm8k.save_to_disk("./raw/math/gsm8k")
    math.save_to_disk("./raw/math/competition_math")

    return gsm8k, math

def download_code_data() -> DatasetDict:
    """Download APPS dataset"""
    print("Downloading APPS...")
    apps: DatasetDict = load_dataset("codeparrot/apps")  # type: ignore

    Path("./raw/code").mkdir(parents=True, exist_ok=True)
    apps.save_to_disk("./raw/code/apps")

    return apps

def filter_and_format_math(dataset, num_samples=5000):
    """
    Filter math problems to a difficulty level suitable for RL.
    Following Magistral's approach: not too easy, not too hard.
    """
    problems = []

    for example in tqdm(dataset['train'][:num_samples]):
        problem = {
            'question': example['question'],
            'answer': example['answer'],
            'type': 'math'
        }
        problems.append(problem)

    Path("./processed").mkdir(exist_ok=True)
    with open("./processed/math_train.jsonl", 'w') as f:
        for p in problems:
            f.write(json.dumps(p) + '\n')

    print(f"Processed {len(problems)} math problems")
    return problems

def filter_and_format_code(dataset, num_samples=2000):
    """
    Filter code problems with test cases.
    """
    problems = []
    for example in tqdm(dataset['train'][:num_samples]):
        if not example.get('input_output'):
            continue
        problem = {
            'question': example['question'],
            'solutions': example['solutions'],
            'input_output': json.loads(example['input_output']),
            'type': 'code'
        }
        problems.append(problem)

    with open("./processed/code_train.jsonl", 'w') as f:
        for p in problems:
            f.write(json.dumps(p) + '\n')

    print(f"Processed {len(problems)} code problems")
    return problems

if __name__ == "__main__":
    gsm8k, math = download_math_data()
    apps = download_code_data()
    math_problems = filter_and_format_math(gsm8k)
    code_problems = filter_and_format_code(apps)
    print(f"Math problems: {len(math_problems)}")
    print(f"Code problems: {len(code_problems)}")

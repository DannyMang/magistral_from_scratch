import re
from typing import Optional
import sympy
from sympy import sympify, simplify, N
from sympy.parsing.latex import parse_latex

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} handling nested braces."""
    start_pattern = r'\\boxed\{'
    match = re.search(start_pattern, text)
    if not match:
        return None

    start_idx = match.end()
    brace_count = 1
    i = start_idx

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start_idx:i-1]

    return None


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace('$', '')
    if answer.lower() in ['yes', 'no', 'true', 'false']:
        return answer.lower()

    try:
        expr = parse_latex(answer)
    except:
        try:
            expr = sympify(answer)
        except:
            return answer.lower()

    try:
        simplified = simplify(expr)
        if simplified.is_number:
            return str(N(simplified, 10))
        else:
            return str(simplified)
    except:
        return str(expr)


def compare_answers(generated: str, ground_truth: str) -> bool:
    gen_norm = normalize_answer(generated)
    truth_norm = normalize_answer(ground_truth)
    if gen_norm == truth_norm:
        return True
    try:
        gen_expr = sympify(gen_norm)
        truth_expr = sympify(truth_norm)
        difference = simplify(gen_expr - truth_expr)
        return difference == 0
    except:
        return gen_norm == truth_norm


def verify_math_answer(
    generation: str,
    ground_truth: str
) -> tuple[bool, float, dict]:
    metadata = {
        'has_format': False,
        'extracted_answer': None,
        'is_correct': False
    }
    extracted = extract_boxed_answer(generation)

    if extracted is None:
        return False, 0.0, metadata

    metadata['has_format'] = True
    metadata['extracted_answer'] = extracted
    is_correct = compare_answers(extracted, ground_truth)
    metadata['is_correct'] = is_correct
    reward = 0.1
    if is_correct:
        reward += 0.9

    return is_correct, reward, metadata

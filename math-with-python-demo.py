import os
import re
import glob
import csv
import shutil
import sys
from vllm import LLM, SamplingParams
from python_executor import PythonExecutor
sys.set_int_max_str_digits(20000)
sys.setrecursionlimit(20000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL = 'models/DeepSeek-R1-Distill-Qwen-7B-Awq'
TEMPERATURE_THINK = 1.0
TEMPERATURE_CODE = 0.25
REPETITION_PENALTY_THINK = 1
REPETITION_PENALTY_CODE = 1

MAX_NUM_SEQS = 24
MAX_TOKENS_THINK = 4096
MAX_TOKENS_CODE = 1024
MAX_ITERATION_CODE = 2
MAX_TOKENS = MAX_TOKENS_THINK + MAX_TOKENS_CODE * MAX_ITERATION_CODE + 1024

LOGITS_BIAS_THINK = {}
LOGITS_BIAS_CODE = {x: -10 for x in [
    3783, 11489, 13824, 14190, 71032,  # wait
    1355,                              # input
    144336,                            # √ -> sqrt
]}

QUESTIONS = []
ANSWERS = []
EVAL_COUNT = 100

DEBUG = True


def load_questions():
    global QUESTIONS, ANSWERS
    for file in glob.glob('dataset/*/reference.csv'):
        with open(file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            problem_index = headers.index('problem')
            answer_index = headers.index('answer')
            for row in reader:
                QUESTIONS.append(row[problem_index])
                ANSWERS.append(int(row[answer_index]))
    return QUESTIONS


def load_model():
    global llm, tokenizer, sampling_params
    llm = LLM(
        MODEL,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_TOKENS,
        trust_remote_code=True,
        seed=1,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        top_p=0.95,
        skip_special_tokens=True,
        max_tokens=MAX_TOKENS,
    )


CODE = """
import itertools
import random
import math
import sympy

def"""

CODE_FOR_ACTUAL_RUN = """
import itertools
import random
import math
import sympy
import sys
sys.set_int_max_str_digits(20000)
sys.setrecursionlimit(20000)
exit=lambda x=0:None

def"""

CODE_PREFIX = """</think>
The problem can be solved by following code.
```python""" + CODE


def question_cleanup(input_string):
    cleaned_string = re.sub(r'\\textit\{(.*?)\}', r'\1', input_string)
    cleaned_string = re.sub(r'\\textbackit\{(.*?)\}', r'\1', cleaned_string)
    return cleaned_string


answer_pattern = re.compile(r'(-?\d+(?:\.\d+)?)')


def parse_answer(output: str) -> int:
    matches = answer_pattern.findall(output)
    if len(matches) == 0:
        return None
    try:
        match = matches[-1]
        if '.' in match:
            match = match.split('.')[0]
            if match[0] == '-':
                return int(float(match % 1000) - 1)
        if len(match) > 5:
            match = '-' + match[-4:] if match[0] == '-' else match[-3:]
        return int(match) % 1000
    except:
        return None


def get_value(res: int) -> float:
    if res == 0:
        return 0.1
    elif res < 10:
        return 0.25
    elif 10 <= res < 1000:
        return 1
    return 0.5


def process(question: str, count: int, max_tokens_think: int, max_tokens_code: int, debug_save_dir="output") -> tuple[int, dict[int, int]]:
    # STEP 1: Think for each question
    prompts = []
    for _ in range(count):
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "Generate a complete Python code to solve the math problem. " +
                question + " The answer should be calculated with modulo 1000, and print to stdout."}
        ]
        starter_text: str = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(starter_text)

    sampling_params.temperature = TEMPERATURE_THINK
    sampling_params.max_tokens = max_tokens_think
    sampling_params.stop = ["</think>"]
    sampling_params.logit_bias = LOGITS_BIAS_THINK
    sampling_params.repetition_penalty = REPETITION_PENALTY_THINK
    llm_outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=not DEBUG,
    )
    if DEBUG:
        os.makedirs(debug_save_dir, exist_ok=True)
        for i, output in enumerate(llm_outputs):
            with open(f"{debug_save_dir}/think_{i}.txt", "w") as f:
                f.write(output.outputs[0].text)

    # STEP 2: Generate code for each question
    sampling_params.temperature = TEMPERATURE_CODE
    sampling_params.max_tokens = max_tokens_code
    sampling_params.stop = ["```"]
    sampling_params.logit_bias = LOGITS_BIAS_CODE
    sampling_params.repetition_penalty = REPETITION_PENALTY_CODE

    for i, output in enumerate(llm_outputs):
        prompts[i] += output.outputs[0].text + CODE_PREFIX

    code_outputs = ["" for _ in range(count)]

    for iter in range(MAX_ITERATION_CODE):
        if DEBUG:
            for i in range(count):
                with open(f"{debug_save_dir}/prompt_{iter}_{i}.txt", "w") as f:
                    f.write(prompts[i])

        llm_outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=not DEBUG,
        )

        codes = []
        for output in llm_outputs:
            codes.append(CODE_FOR_ACTUAL_RUN + output.outputs[0].text)
        if DEBUG:
            for i, code in enumerate(codes):
                with open(f"{debug_save_dir}/code_{iter}_{i}.py", "w") as f:
                    f.write(code)

        executor = PythonExecutor()
        execution_results = executor.batch_apply(codes)

        if DEBUG:
            for i, (res, report) in enumerate(execution_results):
                print(f"Iteration {iter} Execution {i}: \n{res=}\n{report=}\n")
                with open(f"{debug_save_dir}/result_{iter}_{i}.txt", "w") as f:
                    f.write(f"{res}\n\n{report}")

        for i, (res, report) in enumerate(execution_results):
            if report == 'Done' and res.strip():
                code_outputs[i] = res
                if iter < MAX_ITERATION_CODE - 1:
                    prompts[i] += f"```\n\nThe output of code: {res}\n\nSo the complete code would be:\n\n```python" + CODE
            elif report == 'Done':
                if iter < MAX_ITERATION_CODE - 1:
                    prompts[i] += f"```\n\nThe output of code is empty.\n\nSo the complete code would be:\n\n```python" + CODE
            else:
                if iter < MAX_ITERATION_CODE - 1:
                    prompts[i] += f"```\n\nBut this code has error: {report}\n\nSo the complete code would be:\n\n```python" + CODE

    counter = {}
    value = {}

    for res in code_outputs:
        res = parse_answer(res)
        if res is not None:
            counter[res] = counter.get(res, 0) + 1
            value[res] = value.get(res, 0) + get_value(res)

    if not counter:
        return None, {}

    _, result = sorted([(v, k) for k, v in value.items()], reverse=True)[0]
    return (result % 1000 + 1000) % 1000, counter


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEMPERATURE_THINK = float(sys.argv[1])
        TEMPERATURE_CODE = float(sys.argv[2])
        DEBUG = False
        print(f"Using {TEMPERATURE_THINK=}, {TEMPERATURE_CODE=}")

    load_model()
    load_questions()

    if DEBUG:
        shutil.rmtree('output', ignore_errors=True)

    correct_count = 0
    confidence = 0
    for index, question in enumerate(QUESTIONS[:EVAL_COUNT]):
        question = question_cleanup(question)
        result, result_dict = process(question, MAX_NUM_SEQS,
                                      MAX_TOKENS_THINK, MAX_TOKENS_CODE, f"output/question_{index}")
        answer = ANSWERS[index]
        print(f"Final {result=}, {answer=}")

        correct_count += 1 if result == answer else 0
        confidence += result_dict.get(answer, 0) / MAX_NUM_SEQS

        print(f"Accuracy: {correct_count}/{index + 1}")
        print(f"Confidence: {confidence / (index + 1)}")

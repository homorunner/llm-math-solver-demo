import os
import re
import glob
import csv
from vllm import LLM, SamplingParams
from python_executor import PythonExecutor
import sys
sys.set_int_max_str_digits(20000)
sys.setrecursionlimit(20000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL = 'models/DeepSeek-R1-Distill-Qwen-7B'
TEMPERATURE = 0.7

MAX_NUM_SEQS = 8
MAX_TOKENS_THINK = 4096
MAX_TOKENS_CODE = 2048
MAX_TOKENS = MAX_TOKENS_THINK + MAX_TOKENS_CODE + 256

LOGITS_PROB_THINK = {x: -100 for x in [3783, ]}

QUESTIONS = [
  "Fred and George take part in a tennis tournament with $4046$ other players. In each round, the players are paired into $2024$ matches. How many ways are there to arrange the first round such that Fred and George do not have to play each other? (Two arrangements for the first round are \\textit{different} if there is a player with a different opponent in the two arrangements.)",
]

DEBUG = True

def load_questions():
  global QUESTIONS
  QUESTIONS = []
  for file in glob.glob('dataset/AIMO-*/reference.csv'):
    with open(file) as f:
      reader = csv.reader(f)
      headers = next(reader)
      problem_index = headers.index('problem')
      for row in reader:
        QUESTIONS.append(row[problem_index])
  return QUESTIONS

def load_model():
  global llm, tokenizer, sampling_params
  llm = LLM(
    MODEL,
    max_num_seqs=MAX_NUM_SEQS*len(QUESTIONS),      # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_TOKENS,                      # Model context length
    trust_remote_code=True,                        # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    # cpu_offload_gb=10,                           # Maximum amount of CPU RAM to offload to the GPU
    seed=1,
  )
  tokenizer = llm.get_tokenizer()

  sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=0.95,
    skip_special_tokens=True,     # whether to skip special tokens in the output
    max_tokens=MAX_TOKENS,
  )

CODE = """
import itertools
import random
import math
import sympy
"""

CODE_FOR_ACTUAL_RUN = """
import itertools
import random
import math
import sympy
import sys
sys.set_int_max_str_digits(20000)
sys.setrecursionlimit(20000)
exit=lambda x=0:None
"""

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
    return int(match)
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

def process(question: str, count: int, max_tokens_think: int, max_tokens_code: int) -> list[str]:
  #### STEP 1: Think for each question
  prompts = []
  for _ in range(count):
    messages: list[dict[str, str]] = [
      {"role": "user", "content": "Generate a complete Python code to solve the math problem. " + question + " The answer should be calculated with modulo 1000, and print to stdout."}
    ]
    starter_text: str = tokenizer.apply_chat_template(
      conversation=messages,
      tokenize=False,
      add_generation_prompt=True
    )
    prompts.append(starter_text)
  sampling_params.max_tokens = max_tokens_think
  sampling_params.stop = ["</think>"]
  llm_outputs = llm.generate(
    prompts=prompts,
    sampling_params=sampling_params,
    use_tqdm=not DEBUG,
  )
  if DEBUG:
    for i, output in enumerate(llm_outputs):
      with open(f"output/think_{i}.txt", "w") as f:
        f.write(output.outputs[0].text)

  #### STEP 2: Generate code for each question
  prompts = []
  for output in llm_outputs:
    prompts.append(output.outputs[0].text + CODE_PREFIX)
  sampling_params.max_tokens = max_tokens_code
  sampling_params.stop = ["```"]
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
      with open(f"output/code_{i}.py", "w") as f:
        f.write(code)

  executor = PythonExecutor()
  execution_results = executor.batch_apply(codes)

  if DEBUG:
    for i, (res, report) in enumerate(execution_results):
      print(f"Execution {i}: \n{res=}\n{report=}\n")

  counter = {}
  
  for res, report in execution_results:
    if report == 'Done':
      res = parse_answer(res)
      if res is not None:
        counter[res] = counter.get(res, 0) + get_value(res)

  if not counter:
    return None
  
  _, result = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
  return (result % 1000 + 1000) % 1000

if __name__ == "__main__":
  load_model()
  load_questions()

  for question in QUESTIONS:
    print(f"Question: {question}")
    question = question_cleanup(question)
    print(f"Cleaned Question: {question}")
    result = process(question, MAX_NUM_SEQS, MAX_TOKENS_THINK, MAX_TOKENS_CODE)
    print(f"Final {result=}")

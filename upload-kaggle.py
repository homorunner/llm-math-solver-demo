import kagglehub
kagglehub.login()

LOCAL_MODEL_DIR = 'models/DeepSeek-R1-Distill-Qwen-7B-Awq'

MODEL_SLUG = 'deepseek-r1-distill-qwen-awq-dvacheng'
VARIATION_SLUG = '7b'

kagglehub.model_upload(
  handle = f"dvacheng/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Quantized with AWQ')

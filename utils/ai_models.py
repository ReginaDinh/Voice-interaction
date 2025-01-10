import string
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
import torch
from peft import PeftModel
from accelerate import Accelerator

class SpeechRecognitionModel:

  def __init__(self, model_id: string, quantization: bool):
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    self.accelerator = Accelerator()
    self.processor = AutoProcessor.from_pretrained(model_id)
    self.config = GenerationConfig.from_pretrained(model_id)

    if quantization == True:
      self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
      self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype="auto", low_cpu_mem_usage=True, quantization_config=self.quantization_config)
    else:
      self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype="auto", low_cpu_mem_usage=True)

    self.model = self.accelerator.prepare(self.model, self.processor.tokenizer)

  def feature_extract(self, audio):
    return self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
   
  def transcribe(self, audio, language):
    features = self.feature_extract(audio)
    input_feat = features.input_features
    att_mask = features.attention_mask
    if language=="en":
      output = self.model[0].generate(input_feat.half().to(self.device), language="en", attention_mask=att_mask, do_sample=True, temperature=0.7, num_beams=5, top_p=0.95, top_k=5)
    elif language=="vi":
      output = self.model[0].generate(input_feat.to(self.device), attention_mask=att_mask, do_sample=True, temperature=0.7, num_beams=5, top_p=0.95, top_k=5)
    return self.model[1].batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

class GPTModel:
  def __init__(self, model_id: string, quantization: bool):
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    self.accelerator = Accelerator()
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantization == True:
      self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
      self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation = "sdpa", low_cpu_mem_usage=True, quantization_config=self.quantization_config)
    else:
      self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation = "sdpa", low_cpu_mem_usage=True)

    self.model = self.accelerator.prepare(self.model, self.tokenizer)

  def generate(self, prompt):
    messages = [
      {"role": "user", "content": prompt},
    ]
    model_inputs = self.model[1].apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
    start_index = model_inputs.shape[-1]
    terminators = [
      self.model[1].eos_token_id,
      self.model[1].convert_tokens_to_ids("<|eot_id|>")
    ]
    generated_ids = self.model[0].generate(model_inputs, do_sample=True,
                      eos_token_id=terminators,
                      max_new_tokens=256, temperature=0.6,
                      top_p=0.9)
    generated_text = self.model[1].decode(generated_ids[0][start_index:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    messages.append({"role": "system", "content": generated_text})
    return generated_text
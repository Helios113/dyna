from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


print(type(model))
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")
generated_ids = model(**model_inputs, output_attentions=True, output_hidden_states=True)
print(type(generated_ids))
print(generated_ids.attentions.shape)
print(len(generated_ids.hidden_states))

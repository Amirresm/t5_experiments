from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5Model.from_pretrained("t5-small")

# input_ids = tokenizer(
#     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# decoder_input_ids = tokenizer(
#     "Studies show that", return_tensors="pt"
# ).input_ids  # Batch size 1

# # forward pass
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# last_hidden_states = outputs.last_hidden_state

# print(tokenizer.decode(last_hidden_states[0]))

tokenizer = T5Tokenizer.from_pretrained("T5-base")
model = T5ForConditionalGeneration.from_pretrained("T5-base")


code = """
what is a large number in python?
"""

input = f"explain: {code}"

inputs = tokenizer.encode(
    input,
    return_tensors="pt",
    # max_length=512,
    truncation=True,
)

output = model.generate(inputs, max_length=80)

summary = tokenizer.decode(output[0])

print("=====Model Summary:")
print(summary)

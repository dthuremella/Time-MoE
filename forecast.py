import torch
from transformers import AutoModelForCausalLM
import pickle

def predict(seqs):
    # normalize seqs
    mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
    normed_seqs = (seqs - mean) / std

    # forecast
    prediction_length = 12
    output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
    normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]

    # inverse normalize
    predictions = normed_predictions * std + mean
    return predictions

with open('getxy.pickle', 'rb') as handle:
    data = pickle.load(handle)

context_length = 8
# seqs = torch.randn(2, context_length)  # tensor shape is [batch_size, context_length]

history = data['x']
history_x = history[:,:,0]
history_y = history[:,:,1]
future = data['y']
future_x = future[:,:,0]
future_y = future[:,:,1]

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)
import pdb; pdb.set_trace()
prediction_x = predict(history_x)
prediction_y = predict(history_y)

errors = torch.sqrt(torch.pow(prediction_x - future_x,2) + torch.pow(prediction_y - future_y,2))

print(torch.mean(errors))

# use it when the flash-attn is available
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)


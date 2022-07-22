# longformer

## train

```
# base env: conda_python3
# running env cudatoolkit=10.0
longfomertrain.ipynb
```

## predict 

```python
from collections import OrderedDict   #导入此模块
base_weights = torch.load(ckpt_pth)['state_dict']
new_state_dict = OrderedDict()

for k, v in base_weights.items():
    #print (k)
    if k=='model.final_logits_bias':
        new_state_dict['final_logits_bias'] = v 
        new_state_dict[k] = v 
    else:
        new_state_dict[k] = v 

        
import torch 
from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = LongformerEncoderDecoderConfig.from_pretrained('longformer-encdec-base-16384')
#config.attention_dropout = self.args.attention_dropout
#config.gradient_checkpointing = self.args.grad_ckpt
config.attention_mode = 'sliding_chunks'

model.model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained('longformer-encdec-base-16384',config = config)

model.load_state_dict(new_state_dict)
#model = torch.load(ckpt_pth,map_location=device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('longformer-encdec-base-16384')

tokenizer.model_max_length = 4000


SAMPLE_TEXT = ' '.join(['Hello world! '] * 200)  # long input document

input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)

attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
attention_mask[input_ids == tokenizer.pad_token_id] = 0
attention_mask[:, 0] = 2 
half_padding_mod = model.config.attention_window[0]
input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
                input_ids, attention_mask, half_padding_mod, tokenizer.pad_token_id)

generated_ids =  model.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=256,
                                            num_beams=1)

generated_str = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)


```
## deploy(todo)
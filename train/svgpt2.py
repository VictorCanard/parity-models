import torch
from transformers import GPT2Model

if __name__ == '__main__':

    model = GPT2Model.from_pretrained('gpt2')
    torch.save(model.state_dict(), "base_model_trained_files/wikitext/gpt2/model.t7")
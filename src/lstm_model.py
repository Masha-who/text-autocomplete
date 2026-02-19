import torch
import torch.nn as nn


class LSTMAutocomplete(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, eos_token_id=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.eos_token_id = eos_token_id

        self.rnn = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)       
        out, _ = self.rnn(emb)        
        logits = self.fc(out)         
        return logits    
    
    def generate(self, prefix_ids, max_new_tokens=30):
        self.eval()

        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)

        device = next(self.parameters()).device
        prefix_ids = prefix_ids.to(device)

        generated = prefix_ids.clone()
        hidden = None

        with torch.no_grad():

            emb = self.embedding(generated)
            out, hidden = self.rnn(emb, hidden)

            for _ in range(max_new_tokens):
                last_hidden = out[:, -1, :]      
                logits = self.fc(last_hidden)   

                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if self.eos_token_id is not None and int(next_token.item()) == int(self.eos_token_id):
                    break

                emb = self.embedding(next_token)
                out, hidden = self.rnn(emb, hidden)

        return generated.squeeze(0)
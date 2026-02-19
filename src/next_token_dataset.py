import torch

from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=20):
        self.samples = []
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        for line in texts:
            # добавляем EOS к текущей строке
            line = line + " " + tokenizer.sep_token

            token_ids = tokenizer.encode(
                line,
                add_special_tokens=False,
                max_length=512,
                truncation=True
            )

            if len(token_ids) < seq_len + 1:
                continue

            # скользящее окно +1
            for i in range(len(token_ids) - seq_len):
                x = token_ids[i : i + seq_len]
                y = token_ids[i + 1 : i + seq_len + 1]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
import torch
import torch.nn as nn

from tqdm import tqdm


class LSTMTrainer:
    def __init__(self, model, vocab_size, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        print("-" * 30)

        self.model = model.to(self.device)
        self.vocab_size = vocab_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)        

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x_batch)  # [B, T, V]
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),  # [B*T, V]
                y_batch.reshape(-1)                   # [B*T]
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)
    
    def evaluate(self, loader):
        self.model.eval()
        sum_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)

                loss = self.criterion(
                    logits.view(-1, self.vocab_size),
                    y_batch.view(-1)
                )

                sum_loss += loss.item()

        return sum_loss / len(loader)
    
    def fit(self, train_loader, val_loader, n_epochs=5):
        for epoch in range(n_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}") 
            print("-" * 30)
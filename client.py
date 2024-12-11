import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from config import cfg
import typing
import time


class Client:
    def __init__(self, id: int, dataloader, ModelClass, log, device="cpu"):
        self.id = id
        self.dataloader = dataloader
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.model = ModelClass().to(device)
        self.opt = AdamW(self.model.parameters(), lr=cfg.learning_rate)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.log = log

    def train(self):
        print(f"Training client {self.id}!")
        self.log.info(f"Training client {self.id}!")

        self.model.train()
        loss_history = []
        for epoch in range(cfg.num_local_epochs):
            epoch_loss = 0
            t1 = time.time()
            for batch in self.dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                
                self.opt.zero_grad()
                
                #Get loss
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()
            epoch_loss /= len(self.dataloader)
            t2 = time.time()
            loss_history.append(epoch_loss)
            print(f"[Client {self.id}][Epoch {epoch}] Training Loss: {epoch_loss:.4f} | Time: {t2-t1:.2f}")
            self.log.info(f"\t[Client {self.id}][Epoch {epoch}] Training Loss: {epoch_loss:.4f} | Time: {t2-t1:.2f}")

        return loss_history

    def __repr__(self):
        return f"ID: [{self.id}] \n Size of data: {len(self.dataloader)} \n\n\n\n"

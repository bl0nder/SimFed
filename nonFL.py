from config import cfg
from typing import List
import torch
import torch.nn as nn
from torch.optim import AdamW
import time 

class NonFLSetup:
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, ModelClass, log, device="cpu"):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.model = ModelClass().to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.opt = AdamW(self.model.parameters(), lr=cfg.learning_rate)
        self.log = log

    def train(self):
        print("Non-FL training started.")
        self.log.info("Non-FL training started.")
        self.model.train()
        val_interval = cfg.val_interval
        
        training_loss_history = []
        val_loss_history = []
        acc_history = []
        for epoch in range(cfg.num_global_epochs):
            epoch_loss = 0

            t1 = time.time()
            for batch in self.train_dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                
                self.opt.zero_grad()
                
                #Get loss
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()
            epoch_loss /= len(self.train_dataloader)
            t2 = time.time()
            training_loss_history.append(epoch_loss)
            # print(f"[NON-FL][Epoch {epoch}] Loss: {epoch_loss:.4f}")

            #Validate training after an interval
            if ((epoch+1) % val_interval == 0):
                acc, loss = self.test(val=True)
                print(f"[Non-FL][Epoch {epoch}] Global Validation Accuracy: {acc*100:.4f} | Global Validation Loss: {loss:.4f}")
                self.log.info(f"[Non-FL][Epoch {epoch}] Global Validation Accuracy: {acc*100:.4f} | Global Validation Loss: {loss:.4f}")
                val_loss_history.append(loss)
                acc_history.append(acc)
            
            print(f"[Non-FL][Epoch {epoch}] Time: {t2-t1}s")
            self.log.info(f"[Non-FL][Epoch {epoch}] Time: {t2-t1}s")
        
        return training_loss_history, val_loss_history, acc_history

    def test(self, val=False):
        dataloader = self.val_dataloader if val else self.test_dataloader
        
        #Metrics
        avg_loss = 0
        avg_acc = 0
        
        self.model.eval()
        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            
            #Get loss 
            logits = self.model(x)
            loss = self.criterion(logits, y)
            avg_loss += loss.item()

            #Get predicted class labels
            preds = torch.argmax(logits, dim=-1)
            acc = torch.sum(preds == y)/len(y)
            avg_acc += acc.item() 
        
        avg_loss /= len(dataloader)
        avg_acc /= len(dataloader)

        return avg_acc, avg_loss

    def __repr__(self):
        return f"num_clients: {len(self.client_list)}\nLength of val dataloader: {len(self.val_dataloader)}\nLength of test dataloader: {len(self.test_dataloader)}"

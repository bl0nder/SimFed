from config import cfg
from typing import List
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class Server:
    def __init__(self, client_list, val_dataloader, test_dataloader, ModelClass, log, device="cpu"):
        self.client_list = client_list
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.global_model = ModelClass().to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.log = log

    def aggregate_local_weights(self, aggregation_algo="fedavg"):
        aggregated_weights = {}

        if aggregation_algo.lower() == "fedavg":
            total_training_samples = 0
            for client in self.client_list:
                total_training_samples += len(client.dataloader) * cfg.batch_size
            
            for client in self.client_list:
                state_dict = client.model.state_dict()
                avg_factor = len(client.dataloader)*cfg.batch_size / total_training_samples
                for key in state_dict:
                    if key not in aggregated_weights:
                        aggregated_weights[key] = avg_factor * state_dict[key]
                    else:
                        aggregated_weights[key] += avg_factor * state_dict[key]
        
        return aggregated_weights

    def train(self):
        print("Training started.")
        self.log.info("Training started.")
        
        self.global_model.train()
        training_loss_history = []
        val_loss_history = []
        acc_history = []
        for epoch in range(cfg.num_global_epochs):
            t1 = time.time()
            # Copy global model weights to local models
            for client in self.client_list:
                client.model.load_state_dict(self.global_model.state_dict())

            # Train local models
            epoch_loss = 0
            for client in self.client_list:
                client_loss_history = client.train()
                avg_client_loss = sum(client_loss_history)/len(client_loss_history)
                epoch_loss += avg_client_loss
            epoch_loss /= cfg.num_clients
            training_loss_history.append(epoch_loss)

            # Aggregate local weights
            aggregated_weights = self.aggregate_local_weights()

            #Update global model
            self.global_model.load_state_dict(aggregated_weights)
            t2 = time.time()

            #Validate training after an interval
            if ((epoch+1) % cfg.val_interval == 0):
                acc, loss = self.test(val=True)
                print(f"[Server][Epoch {epoch}] Global Validation Accuracy: {acc*100:.4f} | Global Validation Loss: {loss:.4f}")
                self.log.info(f"[Server][Epoch {epoch}] Global Validation Accuracy: {acc*100:.4f} | Global Validation Loss: {loss:.4f}")
                val_loss_history.append(loss)
                acc_history.append(acc)
            
            print(f"[Server][Epoch {epoch}] Time: {t2-t1:.2f}s")
            self.log.info(f"[Server][Epoch {epoch}] Time: {t2-t1:.2f}s")
        
        return training_loss_history, val_loss_history, acc_history


    def test(self, val=False):
        dataloader = self.val_dataloader if val else self.test_dataloader
        
        #Metrics
        avg_loss = 0
        avg_acc = 0
        
        self.global_model.eval()
        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            
            #Get loss 
            logits = self.global_model(x)
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

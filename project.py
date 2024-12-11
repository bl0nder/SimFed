import torch
from torch.optim import AdamW
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from config import cfg
from client import Client
from server import Server
import typing
from typing import List
from model import MNISTModel
from nonFL import NonFLSetup
import logging
from datetime import datetime
import time
import matplotlib.pyplot as plt


######################################################
#               Debugging Functions
def get_data_stats(dataloader):
    images, labels = next(iter(dataloader))
    images = images.squeeze(1)
    print(f"Length: {len(dataloader)}")
    print("Images:", images, images.shape)
    print("Labels:", labels, labels.shape)


######################################################


def create_clients(dataset: Dataset, log) -> List[Client]:
    # Split dataset into disjoint shards
    gen = torch.Generator().manual_seed(42)
    data_shards = random_split(
        dataset=dataset,
        lengths=[len(dataset) // cfg.num_clients] * cfg.num_clients
        + [len(dataset) % cfg.num_clients],
        generator=gen,
    )[:-1]

    # Create client dataloaders
    client_dataloaders = [
        DataLoader(d, batch_size=cfg.batch_size, shuffle=True) for d in data_shards
    ]

    # Create client objects with each shard as client data
    client_list = [
        Client(id, client_dataloaders[id], MNISTModel, log=log, device="cuda")
        for id in range(cfg.num_clients)
    ]

    return client_list


def load_data(name: str = "mnist") -> List[Dataset]:

    if name.lower() == "mnist":
        # Set generator for reproducibility
        train_val_split_generator = torch.Generator().manual_seed(42)

        # Transform for image (also converts PIL image to Torch tensor)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Load datasets and split properly
        dataset: Dataset = MNIST(
            root=".\\Dataset", train=True, transform=transform, download=True
        )
        train_dataset, val_dataset = random_split(
            dataset=dataset, lengths=[50000, 10000], generator=train_val_split_generator
        )
        test_dataset: Dataset = MNIST(
            root=".\\Dataset", train=False, transform=transform, download=True
        )

    return train_dataset, val_dataset, test_dataset


def main():
    # Set up logger
    logfile = datetime.now().strftime(
        "runlog_%H_%M_%d_%m_%Y.log"
    )  # Make new log file for each run
    logging.basicConfig(
        level=logging.INFO,
        filename=f"logs/{logfile}",
        encoding="utf-8",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    log = logging.getLogger(__name__)


    # Get datasets
    train_dataset, val_dataset, test_dataset = load_data()
    
    #Vary number of clients and run experiments
    num_clients = [2,5]
    num_local_epochs = [2,5,10]

    fl_training_loss_histories = []
    fl_val_loss_histories = []
    fl_val_acc_histories = []
    fl_test_acc_history = []
    for n in num_clients:
        cfg.num_clients = n
    
        # Log some important info
        log.info(
            f"""\n\n
            Number of clients: {cfg.num_clients},
            Number of global epochs (communication rounds): {cfg.num_global_epochs},
            Num of local epochs: {cfg.num_local_epochs}
            Batch size: {cfg.batch_size}
            Learning rate: {cfg.learning_rate}
            """
        )

        # Create clients
        client_list = create_clients(train_dataset, log)

        # Create server
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
        server = Server(
            client_list=client_list,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            ModelClass=MNISTModel,
            log=log,
            device="cuda",
        )

        #FL
        server_training_loss_history, server_val_loss_history, server_acc_history = (
            server.train()
        )
        fl_training_loss_histories.append(server_training_loss_history)
        fl_val_loss_histories.append(server_val_loss_history)
        fl_val_acc_histories.append(server_acc_history)
        t1 = time.time()
        acc, loss = server.test()
        t2 = time.time()
        print(
            f"[Server][Test] Global Accuracy: {acc*100:.4f} | Global Loss: {loss:.4f} | Time: {t2-t1:.2f}s"
        )
        log.info(
            f"[Server][Test] Global Accuracy: {acc*100:.4f} | Global Loss: {loss:.4f} | Time: {t2-t1:.2f}s"
        )
        fl_test_acc_history.append(acc)

        #NON-FL
        train_dataloader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True
        )
        non_FL_setup = NonFLSetup(
            train_dataloader,
            val_dataloader,
            test_dataloader,
            MNISTModel,
            log=log,
            device="cuda",
        )
        non_FL_training_loss_history, non_FL_val_loss_history, non_FL_acc_history = (
            non_FL_setup.train()
        )
        t1 = time.time()
        acc, loss = non_FL_setup.test()
        t2 = time.time()
        print(
            f"[Non-FL][Test] Global Accuracy: {acc*100:.4f} | Global Loss: {loss:.4f} | Time: {t2-t1:.2f}"
        )
        log.info(
            f"[Non-FL][Test] Global Accuracy: {acc*100:.4f} | Global Loss: {loss:.4f} | Time: {t2-t1:.2f}"
        )

        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            server_training_loss_history,
            label=f"FL [K={cfg.num_clients}, R={cfg.num_global_epochs}, E={cfg.num_local_epochs}] ",
            color="red",
        )
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            non_FL_training_loss_history,
            label=f"Centralized [E={cfg.num_global_epochs}]",
            color="blue",
        )
        plt.title("Training Losses of FL v/s Centralized Frameworks")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig(f"plots/train_loss_K_{cfg.num_global_epochs}" + logfile[6:-4] + ".png")

        # Plot validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            server_val_loss_history,
            label=f"FL [K={cfg.num_clients}, R={cfg.num_global_epochs}, E={cfg.num_local_epochs}] ",
            color="red",
        )
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            non_FL_val_loss_history,
            label=f"Centralized [E={cfg.num_global_epochs}]",
            color="blue",
        )
        plt.title("Validation Losses of FL v/s Centralized Frameworks")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig(f"plots/val_loss_K_{cfg.num_global_epochs}" + logfile[6:-4] + ".png")

        #Plot accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            server_acc_history,
            label=f"FL [K={cfg.num_clients}, R={cfg.num_global_epochs}, E={cfg.num_local_epochs}] ",
            color="red",
        )
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            non_FL_acc_history,
            label=f"Centralized [E={cfg.num_global_epochs}]",
            color="blue",
        )
        plt.title("Accuracies of FL v/s Centralized Frameworks")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig(f"plots/acc_K_{cfg.num_global_epochs}" + logfile[6:-4] + ".png")

    #Train Loss
    plt.figure(figsize=(10, 5))
    colors = ["red", "blue", "green", "black"]
    for idx, lh in enumerate(fl_training_loss_histories):
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            lh,
            label=f"{num_clients[idx]} clients",
            color=colors[idx],
        )
    plt.title("Training Loss with Varying Number of Clients")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/train_loss_diff_clients" + logfile[6:-4] + ".png")

    #Val Loss
    plt.figure(figsize=(10, 5))
    for idx, lh in enumerate(fl_val_loss_histories):
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            lh,
            label=f"{num_clients[idx]} clients",
            color=colors[idx],
        )
    plt.title("Validation Loss with Varying Number of Clients")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/val_loss_diff_clients" + logfile[6:-4] + ".png")

    #Val Acc
    plt.figure(figsize=(10, 5))
    for idx, lh in enumerate(fl_val_acc_histories):
        plt.plot(
            range(1, cfg.num_global_epochs + 1),
            lh,
            label=f"{num_clients[idx]} clients",
            color=colors[idx],
        )
    plt.title("Validation Accuracy with Varying Number of Clients")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/val_acc_diff_clients" + logfile[6:-4] + ".png") 

    #Test Acc
    plt.figure(figsize=(10, 5))
    plt.plot(
        num_clients,
        fl_test_acc_history,
        color="red"
    )
    plt.title("Testing Accuracy v/s Number of Clients")
    plt.xlabel("Number of Clients")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"plots/test_acc_vs_clients" + logfile[6:-4] + ".png") 

if __name__ == "__main__":
    main()

class Config:
    def __init__(self):
        self.num_local_epochs = 10
        self.num_global_epochs = 10
        self.num_clients = 10
        self.batch_size = 64
        self.learning_rate = 0.005
        self.val_interval = 1

cfg = Config()
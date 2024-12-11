# SimFed 👾
A simple from-scratch implementation of a framework that simulates a federated learning environment to train ML models. It simulates client objects using disjoint data shards and has a server object controlling the training of the global model. Currently, the data shards are created from the MNIST dataset although I do want to add the option to use other, more diverse, datasets as well in the future.


### Instructions to Run Code
Make sure the file structure looks like this:

```
SimFed
├── Dataset
│   ├── MNIST
│   |   ├── raw          
├── logs
│   ├── *.log
├── plots
│   ├── *.png 
├── client.py
├── config.py
├── model.py
├── nonFL.py
├── project.py
├── README.md
├── requirements.txt
└── server.py
```

Install dependencies in requirements.txt using 
```
pip install -r requirements.txt
```

Run code using:
```
python project.py
```

All hyperparameters are in config.py if you want to change them and experiment.
For each run of the program, a log will be created in ./logs. The naming convention for the logs is `runlog_[hour]_[minute]_[day]_[month]_[year].log`. The corresponding plots can be found in a similar way.

Note that the code will train the models for two different numbers of clients (2 and 5) so that it can create some plots for comparison. In case you do not want that, change line 100 in `project.py` to
```
num_clients = [cfg.num_clients]
```
and re-run. In this case, however, it will only create empty plots (since no hyperparameter is changing) but you can check the results from the logs.

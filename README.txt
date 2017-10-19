Steps to run DeepHawkes:

1.split the data to train set, validation set and test set.
command: 
cd gen_sequence
python gen_sequence.py
#you can configure some parameters and filepath in the file of "config.py"

2.initialize the embeddings of nodes.
command:
cd gen_sequence
python gen_NodeEmbedding.py
#similar, you can configure parameters and filepath in the file of "config.py"

3.trainsform the datasets to the format of ".pkl"
command:
cd deep_learning
python preprocess.py
#you can configure some parameters and filepath in the file of "config.py"

4.train DeepHawkes
command:
cd deep_learning
python run_sparse.py learning_rate learning_rate_for_embeddings l2 dropout
#exsamples	python -u run_sparse.py 0.001 0.001 0.001 1.0

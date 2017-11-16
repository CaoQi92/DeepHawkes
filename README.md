DeepHawkes
===================================
This repository is an implementation of our proposed DeepHawkes model in the following paper:
 
    Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, Xueqi Cheng. 2017. DeepHawkes: Bridging the Gap between 
    Prediction and Understanding of Information Cascades. In Proceedings of CIKM'17, Singapore., November 
    6-10, 2017, 11 pages.
 
For more details, you can download this paper from ACM DIGITAL LIBRARY.
The following url is the corresponding download link:
https://dl.acm.org/citation.cfm?id=3132973&CFID=1005695721&CFTOKEN=57128415
 

DataSet
----------------------------------- 
We publish the Sina Weibo Dataset used in our paper,i.e., dataset_weibo.txt. It contains 119,313 messages in June 1, 2016.
Each line contains the information of a certain message, the format of which is:
    
    <message_id>\tab<user_id>\tab<publish_time>\tab<retweet_number>\tab<retweets>
    <message_id>:     the unique id of each message, ranging from 1 to 119,313.
    <root_user_id>:   the unique id of root user. The user id ranges from 1 to 6,738,040.
    <publish_time>:   the publish time of this message, recorded as unix timestamp.
    <retweet_number>: the total number of retweets of this message within 24 hours.
    <retweets>:       the retweets of this message, each retweet is split by " ". Within each retweet, it records 
    the entile path for this retweet, the format of which is <user1>/<user2>/......<user n>:<retweet_time>.
    
This dataset is limited to only use in research. And when you use this dataset, please cite our paper as listed above.

Downlowd link: https://pan.baidu.com/s/1c2rnvJq 

password: ijp6

                                                                                                                                                               
Steps to run DeepHawkes
----------------------------------- 

1.split the data to train set, validation set and test set.
command: 

    cd gen_sequence
    python gen_sequence.py
    #you can configure parameters and filepath in the file of "config.py"
 
2.trainsform the datasets to the format of ".pkl"
command:

    cd deep_learning
    python preprocess.py
    #you can configure parameters and filepath in the file of "config.py"
 
3.train DeepHawkes
command:

    cd deep_learning
    python run_sparse.py learning_rate learning_rate_for_embeddings l2 dropout
    #exsamples  python -u run_sparse.py 0.005 0.0005 0.05 0.8

import numpy as np
import six.moves.cPickle as pickle
import config
LABEL_NUM = 0
#LABEL_NUM = 2
#NUM_SEQUENCE = 1000

# trans the original ids to 0~n-1
class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]

    def length(self):
        return len(self.new_to_original)


#trainsform the sequence to list
def sequence2list(flename):
    graphs = {}
    # with open(DATA_PATH + 'random_walks_train.txt', 'r') as f:
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = []
#            for i in range(1, min(len(walks),NUM_SEQUENCE)):
	    for i in range(1, len(walks)):
                s = walks[i].split(":")[0]
                t = walks[i].split(":")[1]
		#print s,flename
                graphs[walks[0]].append([[int(xx) for xx in s.split(",")],int(t)])
            # print "graph",graphs[walks[0]]
                # graphs[walks[0]].append([int(x.split(":")[0]) for x in walks[i].split()])
    return graphs
#read label and size from cascade file
def read_labelANDsize(filename):
    labels = {}
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            profile = line.split('\t')
            labels[profile[0]] = profile[-1]
            sizes[profile[0]] = int(profile[3])
    return labels,sizes

def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            # print graph,walk
            # print "walk",walk[0],walk[1]
            for i in walk[0]:
                original_ids.add(i)
    print "length of original isd:",len(original_ids)
    return original_ids
def write_XYSIZE_data(graphs,labels,sizes,LEN_SEQUENCE,NUM_SEQUENCE,index,filename):
    #get the x,y,and size  data
    blank_template = []
    for i in range(LEN_SEQUENCE):
        blank_template.append(index.new(-1))
    print "blank_template",len(blank_template),blank_template
    x_data = []
    y_data = []
    sz_data = []
    time_data = []
    rnn_index = []
    for key,graph in graphs.items():
        # print key
        label = labels[key].split()
        y = int(label[LABEL_NUM])
        temp = []
        temp_time = []
        temp_index = []
        count = 0
    	size_temp = len(graph)
    	if size_temp !=  sizes[key]:
        	print size_temp,sizes[key]
        for walk in graph:
            # print walk
            temp_walk = []
            walk_time = walk[1]
            temp_time.append(walk_time)
            temp_index.append(len(walk[0]))
            # print walk
            for w in walk[0]:
                temp_walk.append(index.new(w))
            while len(temp_walk) <LEN_SEQUENCE:
                temp_walk.append(index.new(-1))
            # print temp_walk
            temp.append(temp_walk)
            count +=1
        x_data.append(temp)
        y_data.append(np.log(y+1.0)/np.log(2.0))
        sz_data.append(size_temp)
        time_data.append(temp_time)
        rnn_index.append(temp_index)
    #print x_data
    print len(x_data),len(x_data[0]),len(x_data[0][0])
    pickle.dump((x_data, y_data, sz_data, time_data,rnn_index,index.length()), open(filename,'w'))

def write_node2vec(index,original_ids,filename_in,filename_out):
    np.random.seed(13)
    with open(filename_in, 'r') as f:
        line = f.readline()
        temp = line.strip().split()
        num_nodes = int(temp[0])
        num_dims = int(temp[1])
        node_vec = np.random.normal(size=(index.length(), num_dims))
        for i in range(num_nodes):
            line = f.readline()
            temp = line.strip().split()
            node_id = int(temp[0])
            if not node_id in original_ids:
                continue
            node_vec[index.new(node_id), :] = np.array([float(temp[j]) for j in range(1, len(temp))])


    pickle.dump(node_vec, open(filename_out,'w'))
def get_maxsize(sizes):
    max_size = 0
    for cascadeID in sizes:
        # print cascadeID,sizes[cascadeID]
        max_size = max(max_size,sizes[cascadeID])
    print "max_size",max_size
    return max_size
def get_max_length(graphs):
    max_overlap = 0
    len_sequence = 0
    max_num = 0
    for cascadeID in graphs:
	max_num = max(max_num,len(graphs[cascadeID]))
        for sequence in graphs[cascadeID]:
            len_sequence = max(len_sequence,len(sequence[0]))
    print "max_num:",max_num
    return len_sequence
if __name__ == "__main__":

    graphs_train = sequence2list(config.shortestpath_train)
    graphs_val = sequence2list(config.shortestpath_val)
    graphs_test = sequence2list(config.shortestpath_test)

    # labels_train ,sizes_train = read_labelANDsize(DATA_PATH+"cascade_train.txt")
    # labels_val , sizes_val = read_labelANDsize(DATA_PATH+"cascade_val.txt")
    # labels_test , sizes_test = read_labelANDsize(DATA_PATH+"cascade_test.txt")
    #
    #
    labels_train ,sizes_train = read_labelANDsize(config.cascade_train)
    labels_val , sizes_val = read_labelANDsize(config.cascade_val)
    labels_test , sizes_test = read_labelANDsize(config.cascade_test)


    NUM_SEQUENCE = max(get_maxsize(sizes_train),get_maxsize(sizes_val),get_maxsize(sizes_test))
    print "Number of sequence:",NUM_SEQUENCE
    NUM_SEQUENCE = (NUM_SEQUENCE/20+1)*20
    print NUM_SEQUENCE

    LEN_SEQUENCE_train = get_max_length(graphs_train)
    LEN_SEQUENCE_val = get_max_length(graphs_val)
    LEN_SEQUENCE_test = get_max_length(graphs_test)
    LEN_SEQUENCE = max(LEN_SEQUENCE_train,LEN_SEQUENCE_val,LEN_SEQUENCE_test)
    print "\nlength of sequence:",LEN_SEQUENCE
    print LEN_SEQUENCE_train,LEN_SEQUENCE_val,LEN_SEQUENCE_test

    #get the total original_ids and tranform the index from 1 ~n-1
    original_ids = get_original_ids(graphs_train)\
                    .union(get_original_ids(graphs_val))\
                    .union(get_original_ids(graphs_test))
    # for i in range(LEN_SEQUENCE+1):
    #     original_ids.add(0-i-1)
    original_ids.add(-1)
    print "lenth of original_ids:",len(original_ids)
    index = IndexDict(original_ids)
    pickle.dump((index.new_to_original), open("idmap.pkl", 'w'))
    #write the x,y,and size  data
    write_XYSIZE_data(graphs_train, labels_train, sizes_train,LEN_SEQUENCE,NUM_SEQUENCE,index, config.train_pkl)
    write_XYSIZE_data(graphs_val, labels_val, sizes_val,LEN_SEQUENCE,NUM_SEQUENCE,index, config.val_pkl)
    write_XYSIZE_data(graphs_test, labels_test, sizes_test,LEN_SEQUENCE,NUM_SEQUENCE,index,config.test_pkl)

    # write the node2vec
    write_node2vec(index, original_ids, config.node2vec, config.node2vec_pkl)

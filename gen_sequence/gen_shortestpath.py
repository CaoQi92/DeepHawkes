import config
import six.moves.cPickle as pickle
import time
def gen_cascade_graph(observation_time,pre_times,filename,filename_ctrain,filename_cval,filename_ctest,filename_strain,filename_sval,filename_stest):
    file = open(filename)
    file_ctrain = open(filename_ctrain,"w")
    file_cval = open(filename_cval,"w")
    file_ctest = open(filename_ctest,"w")
    file_strain = open(filename_strain,"w")
    file_sval = open(filename_sval,"w")
    file_stest = open(filename_stest,"w")
    cascades_total = dict()
    for line in file:
        parts = line.split("\t")
        if len(parts) != 5:
            print 'wrong format!'
            continue
        cascadeID = parts[0]
        #print cascadeID
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes !=len(path):
            print  'wrong number of nodes',n_nodes,len(path)
        msg_time = long(parts[2])
        hour = time.strftime("%H",time.localtime(msg_time))
        hour = int(hour)
        #print msg_time,hour
        if hour <=7 or hour >=19 :
			continue
        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            nodes_ok = True
            for n in nodes:
                if int(n)==-1:
                    nodes_ok = False
            if not(nodes_ok):
                print nodes
                continue
            # print nodes
            time_now = int(p.split(":")[1])
            if time_now <observation_time:
                observation_path.append(",".join(nodes)+":"+str(time_now))
                for i in range(1,len(nodes)):
                    edges.add(nodes[i-1]+":"+nodes[i]+":1")
            for i in range(len(pre_times)):
                if time_now <pre_times[i]:
                    labels[i] +=1
        if len(observation_path) <10 or len(observation_path) >1000:
            continue
        cascades_total[cascadeID] =msg_time
	
    n_total = len(cascades_total) 
    print 'total:',n_total
    cascades_type = dict()
    sorted_msg_time = sorted(cascades_total.items(),lambda x,y:cmp(x[1],y[1]))
    count = 0
    for (k,v) in sorted_msg_time:
	if count < n_total*1.0/20*14:
		cascades_type[k] = 1
	elif count <n_total*1.0/20*17:
		cascades_type[k] = 2
	else:
		cascades_type[k] = 3
	count +=1

    

    file.close()
    file = open(filename,"r")
    for line in file:
        parts = line.split("\t")
        if len(parts) != 5:
            print 'wrong format!'
            continue
        cascadeID = parts[0]
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes !=len(path):
            print  'wrong number of nodes',n_nodes,len(path)
        msg_time = time.localtime(long(parts[2]))
        #print msg_time
        hour = time.strftime("%H",msg_time)
        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            nodes_ok = True
            for n in nodes:
                if int(n) ==-1:
                    nodes_ok = False
            if not(nodes_ok):
                print nodes
                continue
            time_now = int(p.split(":")[1])
            if time_now <observation_time:
                observation_path.append(",".join(nodes)+":"+str(time_now))
                for i in range(1,len(nodes)):
                    edges.add(nodes[i-1]+":"+nodes[i]+":1")
            for i in range(len(pre_times)):
                # print time,pre_times[i]
                if time_now <pre_times[i]:
                    labels[i] +=1
        for i in range(len(labels)):
			labels[i] = str(labels[i]-len(observation_path))
        hour = int(hour)
        if cascadeID in cascades_type and cascades_type[cascadeID] == 1:
            file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctrain.write(cascadeID+"\t"+parts[1]+"\t"+parts[2]+"\t"+str(len(observation_path))+"\t"+" ".join(edges)+"\t"+" ".join(labels)+"\n")
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 2:
            file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_cval.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 3:
            file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctest.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")

    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()
if __name__ =="__main__":
    print 'yes'
    observation_time = config.observation_time
    pre_times = [24*3600]
    gen_cascade_graph(observation_time, pre_times, config.cascades, config.cascade_train, config.cascade_val,config.cascade_test,
                      config.shortestpath_train, config.shortestpath_val, config.shortestpath_test)

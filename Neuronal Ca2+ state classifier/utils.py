import csv
import numpy as np
import torch
import torch.utils.data as utils

def load_data(path):
    """
    return:
        edges:[num_graphs, 2, num_edges]
        node_features:[num_graphs, num_nodes, 3]
        labels:[num_graphs]
    """
    # 读取边信息
    edges = []
    tmp_edges = []
    with open(path+'/edges.csv') as edgefile:
        cur_graph = '1'
        edge_reader = csv.reader(edgefile)
        next(edge_reader)
        for row in edge_reader:
            graph_id, src, dst, feat = row
            if graph_id != cur_graph:
                edges.append(tmp_edges)
                cur_graph = graph_id
                tmp_edges = []
            
            tmp_edges.append([int(src)-1,int(dst)-1])
        edges.append(tmp_edges)
    edges = np.array(edges).transpose(0,2,1)

    # 读取节点特征
    node_features = []
    tmp_nodes = []
    with open(path+'/nodes.csv') as nodefile:
        cur_graph = '1'
        node_reader = csv.reader(nodefile)
        next(node_reader)
        for row in node_reader:
            graph_id, node_id, feat = row
            if graph_id != cur_graph:
                node_features.append(tmp_nodes)
                cur_graph = graph_id
                tmp_nodes = []
            feat = feat.split(',')
            feat = [float(x) for x in feat]
            tmp_nodes.append(feat)
        node_features.append(tmp_nodes)
    node_features = np.array(node_features)

    # 读取图标签
    labels = []
    with open(path+'/graphs.csv') as graphfile:
        graph_reader = csv.reader(graphfile)
        next(graph_reader)
        for row in graph_reader:
            labels.append(int(row[2])-1)

    labels = np.array(labels)
    # 数据增强
    # num_graphs = labels.shape[0]
    # num_edges = edges.shape[2]
    # edges = np.repeat(edges,4,axis=0)
    # labels = np.repeat(labels,4,axis=0)
    # node_features = np.repeat(node_features,4,axis=0)
    # node_features[num_graphs:,:,:]=np.random.normal(1,0.5,(num_graphs*3,1,1))*node_features[num_graphs:,:,:]
    # node_features[num_graphs:num_graphs*2,:,:]=
    # edges[num_graphs*2:num_graphs*3,:,num_edges//2:]=


    
    return edges, node_features, labels

def get_dataset(data_path, train_propotion = 0.6, valid_propotion = 0.2, BATCH_SIZE = 1):
    edges, node_features, labels = load_data(data_path)
    # print(node_features.shape, edges.shape,labels.shape)
    # split dataset
    sample_size = labels.shape[0]
    index = np.arange(sample_size-1, dtype = int)
    # print(index)
    np.random.seed(0)
    np.random.shuffle(index)
    train_index = index[0 : int(np.floor(sample_size * train_propotion))]
    valid_index = index[int(np.floor(sample_size * train_propotion)) : int(np.floor(sample_size * (train_propotion + valid_propotion)))]
    test_index = index[int(np.floor(sample_size * (train_propotion + valid_propotion))):]
    # print(train_index)

    train_data, train_label, train_edge = node_features[train_index,:,:], labels[train_index], edges[train_index]
    valid_data, valid_label, valid_edge = node_features[valid_index,:,:], labels[valid_index], edges[valid_index]
    test_data, test_label, test_edge = node_features[test_index,:,:], labels[test_index], edges[test_index]
    # print(train_data.shape, train_label.shape)

    num_graphs = train_label.shape[0]
    # num_edges = edges.shape[2]
    train_edge = np.repeat(train_edge,4,axis=0)
    train_label = np.repeat(train_label,4,axis=0)
    train_data = np.repeat(train_data,4,axis=0)
    train_data[num_graphs:,:,:]=np.random.normal(1,0.5,(num_graphs*3,1,1))*train_data[num_graphs:,:,:]

    train_data, train_label, train_edge = torch.Tensor(train_data), torch.LongTensor(train_label), torch.LongTensor(train_edge)
    valid_data, valid_label, valid_edge = torch.Tensor(valid_data), torch.LongTensor(valid_label), torch.LongTensor(valid_edge)
    test_data, test_label, test_edge = torch.Tensor(test_data), torch.LongTensor(test_label), torch.LongTensor(test_edge)

    train_dataset = utils.TensorDataset(train_data, train_label, train_edge)
    valid_dataset = utils.TensorDataset(valid_data, valid_label, valid_edge)
    test_dataset = utils.TensorDataset(test_data, test_label, test_edge)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)

    return train_dataloader, valid_dataloader, test_dataloader


def pred_dataset(data_path, BATCH_SIZE = 1):
    edges, node_features, labels = load_data(data_path)

    pred_data, pred_label, pred_edge = node_features, labels, edges

    pred_data, pred_label, pred_edge = torch.Tensor(pred_data), torch.LongTensor(pred_label), torch.LongTensor(pred_edge)

    pred_dataset = utils.TensorDataset(pred_data, pred_label, pred_edge)

    pred_dataloader = utils.DataLoader(pred_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)

    return pred_dataloader
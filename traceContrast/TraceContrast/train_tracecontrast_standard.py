import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from TraceContrast_model import TraceContrast # TraceContrast
# from TraceContrast_model_change_pool_v2 import TraceContrast # change pooling
# from ts2vec_add_noise import TS2Vec 
# from ts2vec import TS2Vec # TS2Vec
import tasks
import datautils_tracecontrast_standard as datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import scipy.io as scio
from Bio.Cluster import kcluster
from sklearn.metrics import silhouette_score

date = '0726'
date_name = f'{date}_scn_standard' # {date}_right
# scn_data_path = f'D:/lab/scn/3d/20220516/2021{date}/INTERP/{date}_processed_POI_modified.mat' #poi 0916_int_7-30 0918_6-29 0922_6-29
# scn_data_path = f'D:/lab/scn/3d/data_0509/2021{date}_non/{date}_hand_not.mat'
# scn_data_path = f'D:/lab/scn/Data_0712/20210602_CT02-11/20210602_data.mat'
# scn_data_path = f'D:/lab/scn/3d/2022{date}/2022{date}_data.mat'
scn_data_path = f'E:/scn/2022{date}/2022{date}_data.mat'

# export point cloud
def export_ply_with_label(out, points, colors):
    with open(out, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex ' + str(points.shape[0]) + '\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            cur_color = colors[i, :]
            f.write('%f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2],
                int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)
            ))

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


if __name__ == '__main__':

    # parse command

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=64, help='The batch size (defaults to 8)') # new data 256, 0726 64
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=16, help='The representation dimension (defaults to 320)') # dimension 16 8 TODO change dimension for test
    parser.add_argument('--max-train-length', type=int, default=10000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs') # default none
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    

    print('Loading data... ', end='')

    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)

    # load SCN data, unsupervised task
    elif args.loader == 'SCN':
        task_type = 'unsupervised'
        train_data = datautils.load_SCN()
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + date_name + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()
    
    # model
    model = TraceContrast(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit( # TODO: without network
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    # print(loss_log) ### add
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        # add unsupervised task
        elif task_type == 'unsupervised': # change
            # get embedding by the model encoder
            train_repr = model.encode(train_data) # TODO: normally train
            # train_repr = train_data # TODO: without network, normally closed

            scio.savemat(f'./{run_dir}/train_repr.mat', {'emb':train_repr})

            from sklearn.preprocessing import normalize
            embeddings = np.reshape(train_repr, (train_repr.shape[0],train_repr.shape[1]*train_repr.shape[2]))
            embeddings = normalize(embeddings)

            # plot embedding
            num_nodes = train_data.shape[0]
            # plt.figure(2)
            # for i in range(num_nodes):
            #     plt.stem(np.arange(embeddings.shape[1]), embeddings[i],markerfmt='D', use_line_collection=True)
            # plt.savefig(f'./{run_dir}/embeddings.png')
            # # plt.show()
            # plt.close(2)

            scio.savemat(f'./{run_dir}/embedding.mat', {'emb':embeddings}) # save embeddings
            # print('Embeddings dimension:', embeddings.shape)

            # get tsne
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            x_tsne = tsne.fit_transform(embeddings)
            x_min, x_max = x_tsne.min(0), x_tsne.max(0)
            x_norm = (x_tsne - x_min) / (x_max - x_min)

            # scio.savemat(f'./{run_dir}/tsne.mat', {'tsne':x_norm})

            # color_set = np.array([[170, 35, 49],[90, 144, 172],[85, 146, 85],[196, 149, 38],[88, 30, 173]]) # TODO

            # unsupervised clustering
            silhouette_score_list = []
            for num_classes in range(1,6): # range(2,7)
                if num_classes == 1:
                    y = np.zeros((embeddings.shape[0],),dtype=int)
                else:
                    # TODO: origin kmeans
                    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings) # 加入init
                    y = kmeans.labels_
                    
                    # TODO: cos kmeans
                    # y, error, nfound = kcluster(embeddings, num_classes, dist='u', npass=100) # cosine distance
                    
                    # silhouette_avg = silhouette_score(embeddings, y, metric = 'cosine') # euclidean / cosine
                    # silhouette_score_list.append(silhouette_avg)

                    # TODO: pearson dist
                    # y, error, nfound = kcluster(embeddings, num_classes, dist='c', npass=100) # pearson distance

                scio.savemat(f'./{run_dir}/class_order_{num_classes}.mat', {'order':y})

                class_mean = np.zeros((num_classes,1))
                for k in range(num_classes):
                    class_order = np.where(y==k)
                    tmp_pos = x_norm[class_order,0]
                    class_mean[k] = np.mean(tmp_pos)
                new_mean = np.argsort(class_mean, axis=0)

                # plot tsne
                plt.figure(figsize=(6, 6))
                for i in range(x_norm.shape[0]):
                    for j in range(num_classes):
                        if new_mean[j]==y[i]:
                            color_order = j
                    plt.scatter(
                        # x_norm[i, 0], x_norm[i, 1], marker='*', color=color_set[color_order]/255.0
                        x_norm[i, 0], x_norm[i, 1], marker='.', color=plt.cm.Set1(color_order)
                    )
                if num_classes==1:
                    plt.title(f'{num_classes} Cluster', size = 20, fontweight='bold')
                else:
                    plt.title(f'{num_classes} Clusters', size = 20, fontweight='bold')
                plt.yticks(fontproperties = 'Arial', size = 20, fontweight='bold')
                plt.xticks(fontproperties = 'Arial', size = 20, fontweight='bold')
                plt.savefig(f'./{run_dir}/tsne_{num_classes}.eps', dpi=400)
                plt.close()

                # scn_data_path = f'D:/lab/scn/3d/20220516/20210916/INTERP/0916_int_7-30.mat'
                # scn_data_path = f'D:/lab/scn/3d/2022{date_name}/{date_name}_left.mat'
                scn_data = scio.loadmat(scn_data_path)
                poi = torch.FloatTensor(scn_data['POI'])

                # generate point cloud data
                points = poi.cpu().numpy()
                colors = np.zeros_like(points)
                for i in range(colors.shape[0]):
                    # colors[i] = np.array(plt.cm.Set1(y[i])[:3])
                    if y[i] < 9:
                        colors[i] = np.array(plt.cm.Set1(y[i])[:3])
                    elif y[i] >= 9 and y[i] < 17:
                        colors[i] = np.array(plt.cm.Set2(y[i]-9)[:3])
                    else:
                        colors[i] = np.array(plt.cm.Set3(y[i]-17)[:3])

                export_ply_with_label(f'./{run_dir}/{num_classes}_clusters.ply', points, colors) # 输出点云
        else:
            assert False

        
        # TODO: find best cluster number
        # score =[i+2 for i,j in enumerate(silhouette_score_list) if j == max(silhouette_score_list)]
        # print(f'Best cluster number: {score}, max silhouette score: {max(silhouette_score_list)}')

        # pkl_save(f'{run_dir}/train_data.pkl', train_data)
        # pkl_save(f'{run_dir}/test_data.pkl', test_data)
        # pkl_save(f'{run_dir}/train_label.pkl', train_labels)
        # pkl_save(f'{run_dir}/test_label.pkl', test_labels)
        # pkl_save(f'{run_dir}/out.pkl', out)
        # pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        # print('Evaluation result:', eval_res)

    print("Finished.")

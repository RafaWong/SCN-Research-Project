import numpy as np
import argparse
import os
import time
import datetime
from TraceContrast_model import TraceContrast
import datautils
from utils import init_dl_program, name_with_datetime
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import scipy.io as scio


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
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('scn_data_path', help='The folder name used to load input data')
    parser.add_argument('task', help='The task name used for datautils')

    # optional
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=64, help='The batch size (defaults to 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=16, help='The representation dimension (defaults to 16)')
    parser.add_argument('--max-train-length', type=int, default=10000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 10000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()
    
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')

    # load SCN data
    train_data, poi = datautils.load_SCN(args.scn_data_path, args.task)

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.task + '_' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()
    
    # model
    model = TraceContrast(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )

    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    # get embedding by the model encoder
    train_repr = model.encode(train_data) # TODO: normally train

    from sklearn.preprocessing import normalize
    embeddings = np.reshape(train_repr, (train_repr.shape[0],train_repr.shape[1]*train_repr.shape[2]))
    embeddings = normalize(embeddings)

    num_nodes = train_data.shape[0]

    scio.savemat(f'./{run_dir}/embedding.mat', {'emb':embeddings}) # save embeddings

    # get tsne
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(embeddings)
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)

    scio.savemat(f'./{run_dir}/tsne.mat', {'tsne':x_norm})

    # unsupervised clustering
    for num_classes in range(1,6):
        if num_classes == 1:
            y = np.zeros((embeddings.shape[0],),dtype=int)
        else:
            kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings)
            y = kmeans.labels_
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

        export_ply_with_label(f'./{run_dir}/{num_classes}_clusters.ply', points, colors)


    print("Finished.")

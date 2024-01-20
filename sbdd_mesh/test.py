import numpy as np
from tsnecuda import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tsne(embeddings, save_dir, save_name):
    tsne = TSNE(n_iter=1000, verbose=1,num_neighbors=32,device=0,learning_rate=1500)
    tsne_result = tsne.fit_transform(embeddings)
    print(tsne_result.shape)

    train_labels = np.load("/mnt/workspace/dlly/ucm3/data/cifar10_train_labels.npy")
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1, title='CIFAR100-Train')

    scatter = ax.scatter(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        c = train_labels,
        cmap=plt.cm.get_cmap('magma'),
        # cmap=plt.cm.get_cmap('Paired'),
        # alpha=0.4,
        s=2)

    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)

    plt.show()
    # fig = sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1])
    # lim = (tsne_result.min()-5, tsne_result.max()+5)
    # fig.set_xlim(lim)
    # fig.set_ylim(lim)
    # fig.set_aspect('equal')
    # fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    #
    # fig = fig.get_figure()
    # fig.tight_layout()
    # fig.savefig(os.path.join(save_dir, save_name + '_tsne_plot.png'))
    # fig.savefig(os.path.join(save_dir, save_name + '_tsne_plot.pdf'))

if __name__ == "__main__":
    embeddings = np.load("/mnt/workspace/srikesh/img_classification/tsne/CBDM-pytorch/CIFAR10/cifar10_train_embeddings.npy")
    plot_tsne(embeddings, "./", "test")

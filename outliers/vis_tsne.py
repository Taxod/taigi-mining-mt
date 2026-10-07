import argparse
import os

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from post_processing import cluster_based, whitening


def visualise_tsne(embs: np.ndarray, plot_file: str, title: str):
    pca = PCA(n_components=50)
    embs = pca.fit_transform(embs)

    tsne = TSNE(n_components=2, random_state=0)
    embs_2d = tsne.fit_transform(embs)

    clrs = sns.color_palette("pastel", 8)
    fig = plt.figure(figsize=(16, 12))
    plt.title(title)

    tx = embs_2d[:, 0]
    ty = embs_2d[:, 1]
    plt.scatter(tx, ty, color=clrs[2])

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def vis_tsne_multi(emb_list: list, label_list: list, plot_file: str, title: str):
    """
    Supports visualizing multiple embeddings with different colors, markers, and a legend.
    """
    all_embs = np.concatenate(emb_list)

    pca = PCA(n_components=50)
    all_embs = pca.fit_transform(all_embs)

    tsne = TSNE(n_components=2, random_state=0)
    embs_2d = tsne.fit_transform(all_embs)

    clrs = sns.color_palette("pastel", max(8, len(emb_list)))
    markers = ['*', 'x', '.', '^', 's', 'p', '+', 'D']

    fig = plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16)

    start_idx = 0
    for i, emb in enumerate(emb_list):
        end_idx = start_idx + len(emb)
        tx = embs_2d[start_idx:end_idx, 0]
        ty = embs_2d[start_idx:end_idx, 1]

        plt.scatter(tx, ty,
                    color=clrs[i % len(clrs)],
                    marker=markers[i % len(markers)],
                    label=label_list[i],
                    alpha=0.8)

        start_idx = end_idx

    plt.legend(fontsize=16, loc='upper right')

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_embs(emb_file, load, do_cbie, do_whiten):
    if load == "torch":
        embs = torch.load(emb_file).numpy()
    elif load == "np":
        embs = np.load(emb_file, allow_pickle=True)

    if do_whiten:
        embs = whitening(embs)
    if do_cbie:  # idk if it's meaningful to do both whitening and cbie, but IF we do both, probably this order
        embs = cluster_based(embs, n_cluster=7, n_pc=12, hidden_size=embs.shape[1])
    return embs


def main(args):
    emb_files = []
    labels = []

    # 1. New mode: using lists of files and labels (for 3 or more languages)
    if args.emb_files:
        emb_files = args.emb_files
        labels = args.labels if args.labels else [f"lang_{i}" for i in range(len(emb_files))]
        if not args.plot_file:
            args.plot_file = f'../plots/multi_tsne{args.append_file_name}.png'

    # 2. Old mode: backward compatibility for the original workflow
    else:
        if not args.plot_file:
            args.plot_file = f'../plots/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/tsne{args.append_file_name}.png'

        if not args.emb_file:
            lang = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng1'
            args.emb_file = f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang}{args.append_file_name}.pt'

        emb_files.append(args.emb_file)
        labels.append(args.lang_or_track if args.lang_or_track else 'lng1')

        if args.parallel_vis:
            if not args.parallel_emb_file:
                lang2 = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng2'
                args.parallel_emb_file = f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang2}{args.append_file_name}.pt'

            emb_files.append(args.parallel_emb_file)
            labels.append('lng2')

    # Load all embeddings dynamically
    emb_list = []
    for f in emb_files:
        print(f"Loading {f}...")
        emb_list.append(load_embs(f, args.load, args.do_cbie, args.do_whiten))

    title = f"t-SNE vis. Whiten: {args.do_whiten} CBIE: {args.do_cbie}"

    # Visualize using the appropriate function
    if len(emb_list) > 1:
        vis_tsne_multi(emb_list, labels, args.plot_file, title)
    else:
        visualise_tsne(emb_list[0], args.plot_file, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualise embeddings from a file with tSNE')
    parser.add_argument('--model', type=str, default="xlm-roberta-base", help="name of the model to be analyzed")
    parser.add_argument('--layer', type=int, default=7, help="which model layer the embeddings are from")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki", "sts", "custom"],
                        help="use embeddings from this dataset")
    parser.add_argument('--append_file_name', type=str, default="", help='to load files à la .._whitened.pt')
    parser.add_argument('--lang_or_track', type=str, default="", help='part of the path')

    # Old arguments
    parser.add_argument('--emb_file', type=str, help="location of the file in question")
    parser.add_argument('--parallel_emb_file', type=str, required=False, help="location of second emb file")

    # New arguments for multiple inputs
    parser.add_argument('--emb_files', nargs='+', help="List of embedding files to visualize together")
    parser.add_argument('--labels', nargs='+', help="List of labels corresponding to the embedding files")

    parser.add_argument('--plot_file', type=str, help="where to save the plot")
    parser.add_argument('--parallel_vis', action='store_true', help='if two (parallel) langs should be in same plot')
    parser.add_argument('--load', type=str, default='torch', choices=["torch", "np"],
                        help="library to use for loading [torch, np]")
    parser.add_argument('--do_cbie', action='store_true', help='try doing cbie before')
    parser.add_argument('--do_whiten', action='store_true', help='try doing whitening before')
    args = parser.parse_args()
    main(args)
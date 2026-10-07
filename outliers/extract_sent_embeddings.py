""" turning the colab file into a script """
import argparse
import os
import shutil
import tarfile
import zipfile

import pandas as pd
import torch
import wget
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
# 新增 SentenceTransformer 的 import
from sentence_transformers import SentenceTransformer

from constants import langs_tatoeba, langs_wiki, lang_dict_3_2
from post_processing import whitening, cluster_based


def read_tatoeba_data(target):
    # read target lang data
    with open(f'../data/tatoeba-parallel/tatoeba.{target}-eng.{target}', 'r', encoding='utf-8') as tgt:
        tgt_sents = [line for line in tgt.read().split("\n") if line]
    # read eng data
    with open(f'../data/tatoeba-parallel/tatoeba.{target}-eng.eng', 'r', encoding='utf-8') as eng:
        eng_sents = [line for line in eng.read().split("\n") if line]
    return tgt_sents, eng_sents


def read_tatoeba_data_task(target):
    """ use the mixed-up order data needed for the tatoeba task """
    target_2 = lang_dict_3_2[target]
    with open(f'../data/tatoeba/{target_2}-en.{target_2}', 'r', encoding='utf-8') as tgt:
        tgt_sents = [line for line in tgt.read().split("\n") if line]
    with open(f'../data/tatoeba/{target_2}-en.en', 'r', encoding='utf-8') as eng:
        eng_sents = [line for line in eng.read().split("\n") if line]
    return tgt_sents, eng_sents


def mean_pooling(token_embeddings, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeds(data, model, tokenizer, args, device):
    # 判斷是否為 SentenceTransformer 模型
    if isinstance(model, SentenceTransformer):
        print(f"Encoding sentences using SentenceTransformers (batch_size={args.batch_size})...")
        embeddings = model.encode(data,
                                  batch_size=args.batch_size,
                                  show_progress_bar=True,
                                  convert_to_tensor=True)
        return embeddings.cpu()

    # 原本 Hugging Face 的處理邏輯
    tgt_embeddings_layer = None
    attention_masks = None

    dataloader = DataLoader(data, batch_size=args.batch_size, drop_last=False)
    max_len = max([len(x) for x in tokenizer(data, padding=True, truncation=True)['input_ids']])

    for batch in tqdm(dataloader):
        encoded = tokenizer(batch, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        encoded.to(device)
        if attention_masks is None:
            attention_masks = encoded['attention_mask'].cpu()
        else:
            attention_masks = torch.cat((attention_masks, encoded['attention_mask'].cpu()))
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
            if tgt_embeddings_layer is None:
                tgt_embeddings_layer = output[2][1:][args.layer].cpu()
            else:
                tgt_embeddings_layer = torch.cat((tgt_embeddings_layer, output[2][1:][args.layer].cpu()))
    return mean_pooling(tgt_embeddings_layer, attention_masks)


def load_model(args, device):
    # 如果參數名稱包含 sentence-transformers，就切換載入模式
    if "sentence-transformers" in args.model:
        print(f"Loading SentenceTransformer model: {args.model}")
        model = SentenceTransformer(args.model)
        model.to(device)
        return model, None
    else:
        print(f"Loading HuggingFace model: {args.model}")
        model = AutoModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.to(device)
        model.eval()
        return model, tokenizer


def sentences_from_two_cols(path):
    col1, col2 = [], []
    with open(path, 'r') as f_in:
        for line in f_in:
            sent1, sent2 = line.strip().split("\t")
            col1.append(sent1)
            col2.append(sent2)
    return col1, col2


def download_bucc(langs):
    os.makedirs('../data/bucc2018/')
    for lg in langs:
        wget.download(f'https://comparable.limsi.fr/bucc2018/bucc2018-{lg}-en.training-gold.tar.bz2',
                      '../data/bucc2018')
        with tarfile.open(f'../data/bucc2018/bucc2018-{lg}-en.training-gold.tar.bz2', 'r:bz2') as tar:
            tar.extractall('../data/')

        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.training.{lg}', f'../data/bucc2018/{lg}-en.test.{lg}')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.training.en', f'../data/bucc2018/{lg}-en.test.en')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.training.gold', f'../data/bucc2018/{lg}-en.test.gold')

        os.remove(f'../data/bucc2018/bucc2018-{lg}-en.training-gold.tar.bz2')

        wget.download(f'https://comparable.limsi.fr/bucc2018/bucc2018-{lg}-en.sample-gold.tar.bz2',
                      '../data/bucc2018')
        with tarfile.open(f'../data/bucc2018/bucc2018-{lg}-en.sample-gold.tar.bz2', 'r:bz2') as tar:
            tar.extractall('../data/')

        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.sample.{lg}', f'../data/bucc2018/{lg}-en.dev.{lg}')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.sample.en', f'../data/bucc2018/{lg}-en.dev.en')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.sample.gold', f'../data/bucc2018/{lg}-en.dev.gold')
        os.remove(f'../data/bucc2018/bucc2018-{lg}-en.sample-gold.tar.bz2')
        os.rmdir(f'../data/bucc2018/{lg}-en/')


def extract_bucc_embeds(args, device, langs, model, tokenizer):
    if not os.path.exists('../data/bucc2018/'):
        download_bucc(langs)

    raise NotImplementedError


def extract_custom_parallel_embeds(args, device, model, tokenizer):
    paths = args.custom_files
    print(f"Reading {len(paths)} parallel files: {paths}")

    corpora = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            corpora.append([line.strip() for line in f if line.strip()])

    lengths = [len(c) for c in corpora]
    assert len(set(lengths)) == 1, f"平行語料的行數必須一致！got {dict(zip(paths, lengths))}"

    model_name = os.path.basename(os.path.normpath(args.model))
    output_dir = f'../embs/custom/{model_name}/{args.layer}/'
    os.makedirs(output_dir, exist_ok=True)

    # one cluster per ~300 sentences, shared across all languages
    n_cluster = max(lengths[0] // 300, 1)

    for i, (path, sents) in enumerate(zip(paths, corpora), start=1):
        tag = f'lang{i}'
        print(f"Extracting embeddings for {tag} ({path})...")
        emb = get_embeds(sents, model, tokenizer, args, device)
        torch.save(emb, f'{output_dir}/{tag}.pt')

        if args.save_whitened:
            print(f"Applying Whitening to {tag}...")
            torch.save(torch.Tensor(whitening(emb.numpy())),
                       f'{output_dir}/{tag}_whitened.pt')

        if args.save_cbie:
            print(f"Applying CBIE to {tag}...")
            torch.save(torch.Tensor(cluster_based(
                emb.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=emb.shape[1])),
                f'{output_dir}/{tag}_cbie.pt')

    with open(f'{output_dir}/files.txt', 'w', encoding='utf-8') as f_out:
        for i, path in enumerate(paths, start=1):
            f_out.write(f'lang{i}\t{path}\n')

    print(f"Saved to {output_dir}")

def main(args):
    try:
        int(args.device)
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    except ValueError:
        device = args.device

    if args.dataset == "tatoeba":
        langs = langs_tatoeba
    elif args.dataset == "wiki":
        langs = langs_wiki
    elif args.dataset == "sts":
        langs = []
    elif args.dataset == 'bucc':
        langs = ['de', 'fr', 'ru', 'zh']
    elif args.dataset == 'custom_parallel':
        langs = []
    else:
        raise ValueError("unknown dataset argument")

    model, tokenizer = load_model(args, device)

    if args.dataset == 'tatoeba':
        extract_tatoeba_embeds(args, device, langs, model, tokenizer)

    elif args.dataset == 'wiki':
        extract_wiki_embeds(args, device, langs, model, tokenizer)

    elif args.dataset == 'sts':
        extract_sts_embeds(args, device, model, tokenizer)

    elif args.dataset == 'bucc':
        extract_bucc_embeds(args, device, langs, model, tokenizer)
    elif args.dataset == 'custom_parallel':
        extract_custom_parallel_embeds(args, device, model, tokenizer)


def download_tatoeba():
    os.makedirs('../data/tatoeba')
    # os.makedirs('../data/tatoeba-parallel')
    os.makedirs('../data/labels/tatoeba')
    wget.download("https://github.com/facebookresearch/LASER/archive/master.zip", "../data/")
    with zipfile.ZipFile('../data/LASER-main.zip', 'r') as zip_ref:
        zip_ref.extractall('../data/')
    os.rename('../data/LASER-main/data/tatoeba/v1/', '../data/tatoeba-parallel/')
    for sl3, sl2 in lang_dict_3_2.items():
        if sl3 == 'eng' or sl3 == 'sun':
            continue
        src_file = f'../data/tatoeba-parallel/tatoeba.{sl3}-eng.{sl3}'
        tgt_file = f'../data/tatoeba-parallel/tatoeba.{sl3}-eng.eng'
        src_out = f'../data/tatoeba/{sl2}-en.{sl2}'
        tgt_out = f'../data/tatoeba/{sl2}-en.en'
        shutil.copy(src_file, src_out)

        tgts = [line.strip() for line in open(tgt_file, 'r', encoding='utf-8')]
        idx = range(len(tgts))
        data = zip(tgts, idx)
        labels_out = open(f'../data/labels/tatoeba/test-{sl2}.tsv', 'w', encoding='utf-8')
        with open(tgt_out, 'w', encoding='utf-8') as ftgt:
            for t, i in sorted(data, key=lambda x: x[0]):
                ftgt.write(f'{t}\n')
                labels_out.write(f'[{i}]\n')
        labels_out.close()
    os.remove('../data/LASER-main.zip')
    shutil.rmtree('../data/LASER-main/')


def extract_tatoeba_embeds(args, device, langs, model, tokenizer):
    if not os.path.isdir('../data/tatoeba'):
        download_tatoeba()
    for lang in langs:
        print(f"Currently processing {lang}...")
        try:
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}')
        except OSError:
            print(f"embeddings for {lang} already exist, skipping")
            continue
        if args.tatoeba_use_task_order:
            tgt, eng = read_tatoeba_data_task(lang)
        else:
            tgt, eng = read_tatoeba_data(lang)
        tgt_embeddings = get_embeds(tgt, model, tokenizer, args, device)
        eng_embeddings = get_embeds(eng, model, tokenizer, args, device)
        torch.save(tgt_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
        torch.save(eng_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng.pt')
        if args.save_whitened:
            tgt_whitened = torch.Tensor(whitening(tgt_embeddings.numpy()))
            eng_whitened = torch.Tensor(whitening(eng_embeddings.numpy()))
            torch.save(tgt_whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_whitened.pt')
            torch.save(eng_whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng_whitened.pt')
        if args.save_cbie:
            n_cluster = max(len(eng) // 300, 1)
            tgt_cbie = torch.Tensor(cluster_based(
                tgt_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=tgt_embeddings.shape[1]))
            eng_cbie = torch.Tensor(cluster_based(
                eng_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=eng_embeddings.shape[1]))
            torch.save(tgt_cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_cbie.pt')
            torch.save(eng_cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng_cbie.pt')
        print(f"Finished saving embeddings for {lang} in model {args.model}.")


def extract_sts_embeds(args, device, model, tokenizer):
    if not os.path.exists('../data/sts'):
        print("Download and extract the STS data.")
        os.makedirs('../data/sts/')
        wget.download("http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.eval.v1.1.zip",
                      '../data/STS_text.zip')
        with zipfile.ZipFile('../data/STS_text.zip', 'r') as zip_ref:
            zip_ref.extractall('../data/sts')

        wget.download("http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.gs.zip", '../data/STS_gt.zip')
        with zipfile.ZipFile('../data/STS_gt.zip', 'r') as zip_ref:
            zip_ref.extractall('../data/sts')

        os.remove('../data/STS_gt.zip')
        os.remove('../data/STS_text.zip')

    def extract_sts_lng(file, track, sentences, lng_id):
        rep = get_embeds(sentences, model, tokenizer, args, device)
        torch.save(rep, f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/{lng_id}.pt')
        if args.save_whitened:
            whitened = torch.Tensor(whitening(rep.numpy()))
            torch.save(whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/{lng_id}_whitened.pt')
        if args.save_cbie:
            n_cluster = max(rep.shape[0] // 300, 1)
            cbie = torch.Tensor(
                cluster_based(rep.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=rep.shape[1]))
            torch.save(cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/{lng_id}_cbie.pt')

    def extract_sts(file, track):
        lng1_sent, lng2_sent = sentences_from_two_cols(f'../data/sts/{file}')
        try:
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/')
        except OSError:
            print(f"embeddings for {track} already exist, skipping")
            return
        extract_sts_lng(file, track, lng1_sent, "lng1")
        extract_sts_lng(file, track, lng2_sent, "lng2")

    print("Extract Track 2 ar-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track2.ar-en.txt', 'track2-ar-en')
    print("Extract Track 4a es-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track4a.es-en.txt', 'track4a-es-en')
    print("Extract Track 4b es-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track4b.es-en.txt', 'track4b-es-en')
    print("Extract Track 6 tr-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track6.tr-en.txt', 'track6-tr-en')
    print(f"Finished saving STS for model {args.model}.")


def extract_wiki_embeds(args, device, langs, model, tokenizer):
    # Loading Wikipedia datasets
    df_su = pd.read_csv('../data/Wikipedia/Sundanese.csv', sep=',')
    df_sw = pd.read_csv('../data/Wikipedia/Swahili.csv', sep=',')
    df_en = pd.read_csv('../data/Wikipedia/English.csv', sep=',')
    df_es = pd.read_csv('../data/Wikipedia/Spanish.csv', sep=',')
    df_ar = pd.read_csv('../data/Wikipedia/Arabic.csv', sep=',')
    df_tr = pd.read_csv('../data/Wikipedia/Turkish.csv', sep=',')
    dfs = [df_ar, df_en, df_es, df_su, df_sw, df_tr]
    for lang, df in zip(langs, dfs):
        print(f"processing {lang}...")
        try:
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/')
        except OSError:
            print(f"embeddings for {lang} already exist, skipping")
            continue
        rep = get_embeds(df['Sentence'].tolist(), model, tokenizer, args, device)
        torch.save(rep, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
        if args.save_whitened:
            whitened = torch.Tensor(whitening(rep.numpy()))
            torch.save(whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_whitened.pt')
        if args.save_cbie:
            n_cluster = max(rep.shape[0] // 300, 1)
            cbie = torch.Tensor(
                cluster_based(rep.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=rep.shape[1]))
            torch.save(cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_cbie.pt')
        print(f"Finished saving embeddings for {lang} in model {args.model}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Establish outlier dimensions.')
    parser.add_argument('--model', type=str, help="name of the model", required=True)
    parser.add_argument('--layer', type=int, default=7, help="which model layer to save embeddings are from")
    parser.add_argument('--device', type=str, default="0", help="which GPU/device to use")
    # parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki", "sts", "bucc"],
    #                     help="which dataset to encode (tatoeba, wiki)")
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'], help='so far only for bucc')
    parser.add_argument('--tatoeba_use_task_order', action='store_true', default=False,
                        help='load tatoeba data with non-parallel order so there is sth to predict')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for encoding')
    parser.add_argument('--save_whitened', action='store_true', help='save embeddings processed with whitening as well')
    parser.add_argument('--save_cbie', action='store_true', help='save embeddings processed with cbie as well')
    parser.add_argument('--dataset', type=str, default="tatoeba",
                        choices=["tatoeba", "wiki", "sts", "bucc", "custom_parallel"],
                        help="which dataset to encode")
    parser.add_argument('--custom_files', type=str, nargs='+',
                        help="Paths to parallel text files (two or more, line-aligned)")
    args = parser.parse_args()
    main(args)
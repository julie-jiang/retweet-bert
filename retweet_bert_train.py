import networkx as nx
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import gensim
import IPython
import sys
from argparse import ArgumentParser
from retweet_bert_models import build_rbert_model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            'profile_data', type=str,
            help='Path to CSV data that contains user ids/screennames and ' +\
                 'profiles with headers "user" and "profile"')
    parser.add_argument(
            'retweet_graph', type=str,
            help='Path to comma-separated (weighted) edgelist data for a directed' +\
                 'graph in the format of u,v[,weight]')
    parser.add_argument(
            '--sampling_type', type=str, default='mult_neg',
            help='Negative sampling strategy, choose mult_neg or one_neg')
    parser.add_argument(
            '--base_model', type=str,
            default='bert-base-nli-stsb-mean-tokens',
            help='SBERT base model type.' + \
                 ' See SBERT documentation https://www.sbert.net/')
    parser.add_argument(
            '--output_dir', type=str, default='./rbert',
            help='Path to output model to.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()
    print(vars(args))
    np.random.seed(args.seed)



    df = pd.read_csv(args.profile_data)
    df = df[~df['profile'].isna()] # remove empty profiles
    G = nx.read_weighted_edgelist(args.retweet_graph)
    # keep only nodes in both df and G
    G = G.subgraph(df['user'].values).copy()
    df = df[df['user'].isin(G)]


    print('len(df)', len(df))
    print(nx.info(G))

    mapping = dict((sn, i) for i, sn in enumerate(df['user'].unique()))
    G = nx.relabel_nodes(G, mapping)

    curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = f'{args.output_dir}/{args.base_model}-{args.sampling_type}-{curr_time}/'
    print('Output path', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    profiles = df['profile'].values
    model = build_rbert_model(
            args.sampling_type, 
            args.base_model, 
            profiles, 
            G, 
            output_path, 
            seed=args.seed,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            eval_steps=args.eval_steps)
    print('Encoding embedings')
    start_time = time.time()
    embeddings = model.encode(profiles, show_progress_bar=True)
    print('Done encoding (%ds)' % (time.time() - start_time))
    print('sbert embedding shape', embeddings.shape)     

    npzfile = f'{output_path}/%s_profile_embeddings.npz'
    np.savez(npzfile,                                                               
             embeddings=embeddings,                                                 
             profiles=profiles,                                         
             users=df['user'].values) 
    print('saved to', npzfile) 

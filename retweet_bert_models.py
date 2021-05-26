import networkx as nx
import numpy as np
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, LoggingHandler,\
         SentencesDataset, InputExample
from sentence_transformers.evaluation import TripletEvaluator, \
        BinaryClassificationEvaluator, SequentialEvaluator
from datetime import datetime
import logging
import IPython
import sys

print('Torch cuda is available', torch.cuda.is_available())
logging.basicConfig(                                                            
    format="%(asctime)s - %(message)s",                                         
    datefmt="%Y-%m-%d %H:%M:%S",                                                
    level=logging.INFO,                                                         
    handlers=[LoggingHandler()],                                                
) 
class ProfilesDataset(Dataset):
    def __init__(self, profiles, edgelist, n_nodes, model):
        # profile is a list of InputExample
        # edgelist is list of pairs of integers (0,..., N)
        # model is a sentence transformer
        self.model = model
        self.profiles = []
        for p in profiles:
            self.profiles.append(self.model.tokenize(p))
        self.edgelist = edgelist
        self.all_nodes = np.arange(n_nodes)
        self.dummy_labels = torch.tensor(np.zeros(1), dtype=int)

    def __getitem__(self, item):
        u, v = self.edgelist[item]
        w = np.random.choice(self.all_nodes, 1)[0]
        
        
        pu, pv, pw = self.profiles[u], self.profiles[v], self.profiles[w]
        return [pu, pv, pw], self.dummy_labels[0]
        
    def __len__(self):
        return len(self.edgelist)

def edges_train_test_split(G):

    edges = np.array(list(G.edges())).astype(int)
    n_test = int(0.1 * len(edges))  # sample n random edges  
    print('n test edges', n_test) 
    edge_idx = np.arange(G.number_of_edges()) 
    rand_idx = np.random.choice(edge_idx, n_test, replace=False)
    test_edges = edges[rand_idx]
    train_edges = edges[np.delete(edge_idx, rand_idx)]
    print('N train edges %d, N testedges %d, N nodes %d' \
          % (len(train_edges), len(test_edges), G.number_of_nodes()))
    assert len(test_edges) + len(train_edges) == len(edges)
    return train_edges, test_edges

def get_evaluators(test_edges, G, profiles, get_binary_eval=False):
    test_set_triplet = []                                                           
    test_set_binary = []                                                            
    all_nodes = np.arange(G.number_of_nodes())                                      
    for u, v in test_edges:                                                         
        w = np.random.choice(all_nodes, 1)[0]                                       
        pu, pv, pw = profiles[u], profiles[v], profiles[w]                          
        test_set_triplet.append(InputExample(texts=[pu, pv, pw]))
        if get_binary_eval:                   
            w2 = np.random.choice(all_nodes, 1)[0]                                          
            pw2 = profiles[w]                                                           
            test_set_binary.append(InputExample(texts=[pu, pv], label=1))               
            if np.random.random_sample() > 0.5:                                         
                pu = pv                                                                 
            test_set_binary.append(InputExample(texts=[pu, pw2], label=0))                
    tripletEvaluator = TripletEvaluator.from_input_examples(                        
            test_set_triplet, name='test_triplet')                                 
    if get_binary_eval: 
        binaryEvaluator = BinaryClassificationEvaluator.from_input_examples(            
                test_set_binary, name='test_binary')                                    
        seq_evaluator = SequentialEvaluator(                                            
                [tripletEvaluator, binaryEvaluator],                                    
                main_score_function=lambda scores: np.mean(scores))
        return seq_evaluator
    else:
        return tripletEvaluator

def rbert_one_neg(model_name, profiles, G):
    model = SentenceTransformer(model_name)
    print('Loading model dataset')
    train_edges, test_edges = edges_train_test_split(G)

    train_dataset = ProfilesDataset(profiles, train_edges, G.number_of_nodes(), model)
    train_loss = losses.TripletLoss(model=model)

    print('prepping evluators')                                                     
    evaluator = get_evaluators(test_edges, G, profiles)
    return model, train_dataset, trian_loss, evaluator

def rbert_mult_neg(model_name, profiles, G):
    model = SentenceTransformer(model_name)
    print('Loading model dataset')
    train_edges, test_edges = edges_train_test_split(G)
    train_set = [InputExample(texts=[profiles[u], profiles[v]], label=0) \
                 for u, v in train_edges]
    train_dataset = SentencesDataset(train_set, model=model)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    print('prepping evluators')                                                     
    evaluator = get_evaluators(test_edges, G, profiles, get_binary_eval=True)
    return model, train_dataset, train_loss, evaluator

def build_rbert_model(sampling_type, base_model, profiles, G, output_path,
                      seed=2020, batch_size=32, num_epochs=2, eval_steps=10000):
    torch.manual_seed(seed)
    if sampling_type == 'one_neg':
        model_func = rbert_one_neg
    elif sampling_type == 'mult_neg':
        model_func = rbert_mult_neg
    else:
        raise ValueError(sampling_type)
    model, train_dataset, train_loss, evaluator = model_func(base_model, profiles, G)
    
    start_time = time.time()
    train_dataloader = DataLoader(train_dataset, 
                                  shuffle=True, 
                                  batch_size=batch_size)
   
    # 10% of train data
    warmup_steps = int(len(train_dataset) * num_epochs / batch_size * 0.1) 
    print('Number of training steps', len(train_dataloader))
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=eval_steps,
        warmup_steps=warmup_steps,
        output_path=output_path
    )
    print('Done training (%ds)' % (time.time() - start_time))
    return model

    

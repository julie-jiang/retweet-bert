# retweet-bert
This repo contains the source code for the Retweet-Bert [paper](https://arxiv.org/abs/2207.08349). If you find it useful, please cite our (to be) published paper:

Julie Jiang, Xiang Ren, and Emilio Ferrara. Retweet-BERT: Characterizing Echo Chambers by Leveraging Language Features and Information Diffusion in Social Media. The 17th International AAAI Conference on Web and Social Media (ICWSM ’23). [Preprint](https://arxiv.org/abs/2207.08349)

### Requirements
- python 3
- Sentence Transformers 0.3.9
- Transformers 3.5.1
- Pytorch 1.7.0

### Data
Due to Twitter data sharing policy, we are unable to share the dataset used in this paper except for the Tweet IDs. You will have to build the dataset by collecting the data themselves. The data used in this paper can be found [here](https://www.notion.so/Personal-66543fb1b1094fc28447d84e489383b0).

The data should come in two files:
- A CSV file with two columns: `user` and `profile` where each row is the profile description of a user. There should be no duplicates. The first row should be the column headers.
- A weighted edgelist with three columns `user1`, `user2` and an integer `weight`. The user id/names must correpond to the same ones from the first CSV file.

### Usage
```
$ python retweet_bert_train.py
usage: retweet_bert_train.py [-h] [--sampling_type SAMPLING_TYPE]
                             [--base_model BASE_MODEL]
                             [--output_dir OUTPUT_DIR]
                             [--batch_size BATCH_SIZE]
                             [--num_epochs NUM_EPOCHS]
                             [--eval_steps EVAL_STEPS] [--seed SEED]
                             profile_data retweet_graph
retweet_bert_train.py: error: the following arguments are required: profile_data, retweet_graph
```

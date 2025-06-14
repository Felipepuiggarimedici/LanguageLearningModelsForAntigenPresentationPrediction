import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from modelAndPerformances import *
#needed for iterating over three-tuple
from itertools import product
import argparse

pep_max_len = 46 # peptide; enc_input max sequence length
hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len
pep_max_len, hla_max_len

vocab = np.load('tokenizer/Transformer_vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)

# Transformer Parameters
d_model = 64  # Embedding Size
patienceMini = 3 #patience for k-fold validation

parser = argparse.ArgumentParser(description="Getting arguments from command line")
parser.add_argument("-nHeads", type=int, required=True, help="Number of heads")
parser.add_argument("-nLayers", type=int, required=True, help="Number of Layers")
parser.add_argument("-d_k", type=float, required=True, help="d_k")
parser.add_argument("-d_ff", type=int, required=True, help="feed forward dimension")
args = parser.parse_args()
nHeads, nLayers, d_k, d_ff = (int(args.nHeads), int(args.nLayers), int(args.d_k), int(args.d_ff))
batch_size_mini = 1024
epochs = 50
threshold = 0.5

folds = 5 #5 Fold validation used throughout

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
results = []

metrics_all = []
for fold in range(folds):
    ys_train_fold_dict, ys_val_fold_dict = {}, {}
    train_fold_metrics_list, val_fold_metrics_list = [], []
    independent_fold_metrics_list, external_fold_metrics_list, ys_independent_fold_dict, ys_external_fold_dict = [], [], {}, {}
    attns_train_fold_dict, attns_val_fold_dict, attns_independent_fold_dict, attns_external_fold_dict = {}, {}, {}, {}
    loss_train_fold_dict, loss_val_fold_dict, loss_independent_fold_dict, loss_external_fold_dict = {}, {}, {}, {}

    print('=====Fold-{}====='.format(fold))
    print('=====Number of Heads: {}, Number of Layers ={}, d_k=d_v: {}, d_ff = {} ====='.format(nHeads, nLayers, d_k, d_ff))
    print('-----Generate data loader-----')
    train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data_with_loader(type_ = 'train', fold = fold,  batch_size = batch_size_mini)
    val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(type_ = 'val', fold = fold,  batch_size = batch_size_mini)

    print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.label), Counter(val_data.label)))

    print('-----Compile model-----')
    model = Transformer(d_model, d_k, nLayers, nHeads, d_ff).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)#, momentum = 0.99)

    print('-----Train-----')

    metric_best, ep_best = 0, -1
    time_train = 0
    patienceCounter = 0
    for epoch in range(1, epochs + 1):

        _, _, _, time_train_ep = train_step(model, train_loader, fold, epoch, epochs, criterion, optimizer, threshold, use_cuda)
        _, _, val_metrics = eval_step(model, val_loader, criterion, threshold, use_cuda= use_cuda)
        avg_val_metric = sum(val_metrics[:4]) / 4

        if avg_val_metric > metric_best:
            metric_best = avg_val_metric
            best_epoch = epoch
            best_model_state = model.state_dict()
            patienceCounter = 0
        else:
            patienceCounter += 1

        if patienceCounter >= patienceMini:
            break

        time_train += time_train_ep

    model.load_state_dict(best_model_state)
    model.eval()
    _, _, val_metrics = eval_step(model, val_loader, criterion, threshold, use_cuda= use_cuda)
    metrics_all.append(val_metrics)

    print('-----Optimization Finished!-----')
    print("Total training time: {:6.2f} sec".format(time_train))
    print('Parameters tested: n_heads: {}, n_layers: {}, d_k {}, fold: {}'.format(nHeads, nLayers, d_k, fold))
metrics_all = np.array(metrics_all)
metrics_mean = metrics_all.mean(axis=0)
results.append({
    'n_heads': nHeads,
    'n_layers': nLayers,
    'd_k': d_k,
    "d_ff": d_ff,
    'roc_auc': metrics_mean[8],
    'accuracy': metrics_mean[1],
    'mcc': metrics_mean[2],
    'f1': metrics_mean[3],
    'sensitivity': metrics_mean[4],
    "specificity": metrics_mean[5],
    "precision":  metrics_mean[6],
    "recall":  metrics_mean[7],
    "aupr":  metrics_mean[0]
}) 

import os
import pandas as pd

import os
import pandas as pd

csv_path = 'hyperparametersMainModel.csv'
new_result_df = pd.DataFrame(results)

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)

    # Set hyperparameters as index for easy matching
    existing_df.set_index(['n_heads', 'n_layers', 'd_k', 'd_ff'], inplace=True)
    new_result_df.set_index(['n_heads', 'n_layers', 'd_k', 'd_ff'], inplace=True)

    # Update existing rows with new results (overwrite)
    existing_df.update(new_result_df)

    # Append new rows that don't exist yet
    combined_df = pd.concat([existing_df, new_result_df.loc[~new_result_df.index.isin(existing_df.index)]])

    # Reset index before filtering
    combined_df.reset_index(inplace=True)
else:
    combined_df = new_result_df

# Drop rows where d_ff is NaN or empty string
combined_df = combined_df[combined_df['d_ff'].notna()]        
combined_df = combined_df[combined_df['d_ff'].astype(str) != ''] 

combined_df.to_csv(csv_path, index=False)
print(f"Updated and saved results in {csv_path}")


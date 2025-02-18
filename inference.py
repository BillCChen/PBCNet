import argparse
import os
import random
import time
from collections import defaultdict

code_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(code_path + '/model_code')

code_path = code_path.rsplit("/", 1)[0]

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_code.train.predict import predict
from model_code.Dataloader.dataloader import collate_fn, LeadOptDataset, LeadOptDataset_test
from model_code.ReadoutModel.readout_bind import DMPNN # the PBCNet
from model_code.utilis.function import get_loss_func
from model_code.utilis.initial import initialize_weights
from model_code.utilis.scalar import StandardScaler
from model_code.utilis.scheduler import NoamLR_shan
from model_code.utilis.trick import Writer,makedirs
from model_code.utilis.utilis import gm_process
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
def freezen(model):
    need_updata = ['FNN.0.weight', 'FNN.0.bias', 'FNN.2.weight', 'FNN.2.bias', 'FNN.4.weight', 'FNN.4.bias', 'FNN.6.weight', 'FNN.6.bias',
                   'FNN2.0.weight', 'FNN2.0.bias', 'FNN2.2.weight', 'FNN2.2.bias', 'FNN2.4.weight', 'FNN2.4.bias', 'FNN2.6.weight', 'FNN2.6.bias']

    for name, parameter in model.named_parameters():
        if name in need_updata:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True



def softmax(x):
    x_exp = np.exp(x)
    # axis=0
    x_sum = np.sum(x_exp, axis=0)
    s = x_exp / x_sum
    return s


def setup_cpu(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






# args: device, loss_function, continue_learning, retrain,batch_size,init_lr, max_lr, final_lr
def train(args, model, test_loader, device):


    #  ===============  Performance evaluation before fine tuning ===================
    # without finetuning ligands
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,_,_ = predict(model, test_loader, device)

    df = pd.read_csv( "/home/chenqixuan/drug_rerank/src/benchmark/model/PBCNet/PBCNet/data/10_reference/fep1_predict_.csv")
    abs_label = np.array(df.Lable.values).astype(float) + np.array(df.Lable2.values).astype(float)
    abs_predict = np.array(valid_prediction).astype(float) + np.array(df.Lable2.values).astype(float)
    label = np.array(df.Lable.values).astype(float)
    prediction = np.array(valid_prediction).astype(float)
    # =================  ranking related indicators ====================
    Ligand1 = df.Ligand1_num.values
    Ligand2 = df.Ligand2_num.values

    _edge_df = pd.DataFrame({"Ligand1": Ligand1, "Ligand2": Ligand2, "label": label, "predict": prediction})
    _df = pd.DataFrame({"Ligand1":Ligand1, "abs_label":abs_label, "abs_predict":abs_predict})
    _df_group = _df.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

    spearman = _df_group[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
    pearson = _df_group[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
    kendall = _df_group[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]

    print(f"without ligands RMSE_G {rmse_g} Spearman {spearman} Pearson {pearson} Kendall {kendall}")
    _edge_df_output = "/home/chenqixuan/drug_rerank/src/benchmark/model/PBCNet/PBCNet/data/output/edge_df.csv"
    _edge_df.to_csv(_edge_df_output)
    _df_output = "/home/chenqixuan/drug_rerank/src/benchmark/model/PBCNet/PBCNet/data/output/abs_df.csv"
    _df.to_csv(_df_output)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('--log_frequency', type=int, default=100, help='Number of batches reporting performance once' )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--loss_function', type=str, default="mse",
                        help='The loss function used to train the model: mse, smoothl1, mve, evidential')
    parser.add_argument("--device", type=int, default=0,
                        help="The number of device")
    parser.add_argument('--patience', type=int, default=10,help='the patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2,help='the random seed' )
    parser.add_argument('--cpu_num', type=int, default=4,help='the number of cpu')
    parser.add_argument('--cs', type=int, default=0,help=("0: close cross attention, 1: open cross attention"))
    parser.add_argument('--two_task', type=int, default=0,help=("Whether to use auxiliary tasks \
                                                                0: just regeresion, 1: regeresion + classfication"))


    # 
    parser.add_argument('--label_scalar', type=int, default=0, help=("0: close scalar, else: open scalar"))
    parser.add_argument('--continue_learning', type=int, default=1, help=("0: close, 1: open"))

    parser.add_argument('--GCN_', type=int, default=0, help=("Whether to use GCN\
                                                              0: close, 1: open"))
    parser.add_argument('--degree_information', type=int, default=1, help=("Whether to use node degree information \
                                                                            0: close, 1: open"))
    # the hyper-parameters for AU-MPNN
    parser.add_argument('--early_stopping_indicator', type=str, default="pearson")
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--p_dropout', type=float, default=0.2)
    parser.add_argument('--ffn_num_laryers', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.0001)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--final_lr', type=float, default=0.0001)

    parser.add_argument('--result_file', type=str, default=6)
    # message passing model
    parser.add_argument('--encoder_type', type=str, default="DMPNN_res",
                        help="DMPNN_res, SIGN")
    parser.add_argument('--readout_type', type=str, default="atomcrossatt_pair")

    # the hyper-parameters for retrain (finetune)
    parser.add_argument('--retrain', type=int, default=0,
                        help='Whether to continue training with an incomplete training model (Finetune)')
    parser.add_argument('--train_path', type=str,
                        default="/home/yujie/leadopt/data/ic50_graph_rmH_new_2/train_1_pair.csv")
    parser.add_argument('--val_path', type=str,
                        default="/home/yujie/leadopt/data/ic50_graph_rmH_new_2/validation_1_pair.csv")

    parser.add_argument('--fold', type=str, default="0.1")
    # parser.add_argument('--finetune_filename', type=str, default="pfkfb3")



    args = parser.parse_args()

    setup_cpu(args.cpu_num)
    setup_seed(args.seed)

    cuda = "cuda:" + str(args.device)
    cuda = 'cpu'


    fep1 = ['PTP1B', 'Thrombin', 'Tyk2', 'CDK2', 'Jnk1', 'Bace', 'MCL1', 'p38']
    fep2 = ['syk', 'shp2','pfkfb3',  'eg5', 'cdk8', 'cmet', 'tnks2', 'hif2a']



    model = torch.load('/home/chenqixuan/drug_rerank/src/benchmark/model/PBCNet/PBCNet/PBCNet.pth',map_location="cpu")
    model.to(cuda)


    prediction_dataset = LeadOptDataset(
        # f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict.csv"
        "/home/chenqixuan/drug_rerank/src/benchmark/model/PBCNet/PBCNet/model_data/10_reference/fep2_predict_.csv"
        )
    prediction_dataloader = GraphDataLoader(prediction_dataset, collate_fn=collate_fn,
                                            batch_size=args.batch_size,
                                            drop_last=False, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)

    train(args=args, model=model,test_loader=prediction_dataloader, device=cuda)
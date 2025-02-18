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
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_code.train.predict import predict
from model_code.Dataloader.dataloader import collate_fn,collate_fn_pkl, LeadOptDataset, LeadOptDataset_test,LeadOptDataset_pkl
from model_code.ReadoutModel.readout_bind import DMPNN # the PBCNet
from model_code.utilis.function import get_loss_func
from model_code.utilis.initial import initialize_weights
from model_code.utilis.scalar import StandardScaler
from model_code.utilis.scheduler import NoamLR_shan
from model_code.utilis.trick import Writer,makedirs
from model_code.utilis.utilis import gm_process

from graph_save_function import block_to_graphs, Featurization_parameters, read_lmdb
from tqdm import tqdm


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


def generate_data(args,data_dir):
    setup_cpu(10)

    PARAMS = Featurization_parameters()

    Interaction_Bond_Build = False
    rmH = True
    knowledge = pd.read_excel("/run_PBCNet/GARF_probability_distribution.xlsx")
    pockets = read_lmdb(args.pocket_lmdb)
    ligands = read_lmdb(args.ligand_lmdb)
    all_data = []
    pro_ligand = []
    g_pockets = []
    pockets_order = [pro["protein"] for pro in pockets]
    for pocket_,ligands_ in tqdm(zip(pockets,ligands),total=len(pockets),desc='generate complex graph data'):
        ligand = []
        labels = []
        print("len(ligands_):",len(ligands_))
        
        for lig in ligands_:
            pdb_file_str,sdf_file_str = pocket_['pocket'],lig['sdf']
            g_complex,g_pock = block_to_graphs(pdb_file_str,sdf_file_str, Interaction_Bond_Build, rmH, knowledge)
            ligand.append(g_complex)
            labels.append(lig['label'])
        pro_ligand.append((ligand,labels))
        g_pockets.append(g_pock)
    print('generate graph data finished')
    for num in range(len(pro_ligand)):
        pro_lig,labels = pro_ligand[num]
        # pro_lig 是一个长度为 len(pro_lig)的列表，现在需要生成一些 pair 数据存入 all_data 中，先随机选取 pro_lig 中的十个配体，然后每一个配体都和所有的配体（对应同一个蛋白质）组成一个 pair
        length = len(pro_lig)
        # 在 0~length-1 中随机选取 10 个数
        assert length > 10
        random_index = random.sample(range(length), 10)
        # print(f'num:{num},length:{length},random_index:{random_index}')
        for i in random_index:
            for j in range(length):
                all_data.append((pro_lig[i], pro_lig[j],g_pockets[num],labels[i]-labels[j],labels[i],labels[j],i,j,length))
    out_file = f'{data_dir}/output_graph.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(all_data, f)
    return pockets_order

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pocket_lmdb', type=str, default='/data_lmdb/pocket.lmdb')
    parser.add_argument('--ligand_lmdb', type=str, default='/data_lmdb/mol.lmdb')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--device", type=int, default=0,
                        help="The number of device")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=666,help='the random seed' )
    parser.add_argument('--cpu_num', type=int, default=4,help='the number of cpu')


    args = parser.parse_args()

    setup_cpu(args.cpu_num)
    setup_seed(args.seed)

    # cuda = "cuda:2"
    device = 'cpu'

    model = torch.load('/run_PBCNet/PBCNet.pth',map_location=device)
    
    # 遍历所有的蛋白质和配体，生成用于计算的数据集list
    pockets_order = generate_data(args,"/data_lmdb")
    prediction_dataset = LeadOptDataset_pkl('/data_lmdb')
    prediction_dataloader = GraphDataLoader(prediction_dataset, collate_fn=collate_fn_pkl,
                                            batch_size=args.batch_size,
                                            drop_last=False, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,label_1,lable_2,rank_list_length = predict(model, prediction_dataloader, device, get_ori_labels=True)
    collect_data = (valid_prediction, valid_labels,label_1,lable_2,rank_list_length)
    # 保存结果
    with open('/data_lmdb/collect_data.pkl', 'wb') as f:
        pickle.dump(collect_data, f)


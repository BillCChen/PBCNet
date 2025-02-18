eval "$(conda shell.bash hook)"
conda activate PBCNet
PYTHONPATH=/run_PBCNet  python -u /run_PBCNet/ranking_inference_lmdb.py \
--pocket_lmdb /data_lmdb/pockets.lmdb \
--ligand_lmdb /data_lmdb/mols.lmdb \
--batch_size 20 



import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset, get_mesh_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
import multiprocessing
from functools import partial


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


def sample_on_gpu(i, gpu_id, test_set, num_samples, batch_size, num_steps, pos_only, center_pos_mode, sample_num_atoms, result_path, model):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = test_set[i]
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, num_samples, batch_size=batch_size, device=device, num_steps=num_steps, pos_only=pos_only, center_pos_mode=center_pos_mode, sample_num_atoms=sample_num_atoms
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list
    }

    result_path = result_path
    os.makedirs(result_path, exist_ok=True)
    # shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{i}.pt'))



def sample(ckpt_path,data_id,result_path):
    num_steps = 1000
    num_samples = 10
    seed=2021
    device = f"cuda:{data_id}"
    sample_num_atoms = "prior"
    center_pos_mode= "protein"
    pos_only = False
    batch_size = 10
    logger = misc.get_logger('sampling')

    misc.seed_all(seed)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    # dataset, subsets = get_dataset(
    #     config=ckpt['config'].data,
    #     transform=transform
    # )

    dataset, subsets = get_mesh_dataset(
        name=ckpt['config'].data_name,
        path=ckpt['config'].data_path,
        split_path=ckpt['config'].data_split,
        transform=transform
    )

    train_set, test_set = subsets['train'], subsets['val']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'],
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    )
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {ckpt_path}')

    # num_gpus = torch.cuda.device_count()
    # processes = []
    # for i in range(8):
    #     gpu_id = i % num_gpus
    #     p = multiprocessing.Process(target=sample_on_gpu, args=(
    #     i, gpu_id, test_set, num_samples, batch_size, num_steps, pos_only, center_pos_mode, sample_num_atoms,
    #     result_path, model))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()

    data = test_set[data_id]
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, num_samples,
        batch_size=batch_size, device=device,
        num_steps=num_steps,
        pos_only=pos_only,
        center_pos_mode=center_pos_mode,
        sample_num_atoms=sample_num_atoms
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list
    }
    logger.info('Sample done!')

    os.makedirs(result_path, exist_ok=True)
    torch.save(result, os.path.join(result_path, f'result_{data_id}.pt'))


    return result_path


def sample_in_one_device(ckpt_path, result_path, device):
    num_steps = 1000
    num_samples = 10
    seed=2021
    sample_num_atoms = "prior"
    center_pos_mode= "protein"
    pos_only = False
    batch_size = 10
    logger = misc.get_logger('sampling')

    misc.seed_all(seed)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    # dataset, subsets = get_dataset(
    #     config=ckpt['config'].data,
    #     transform=transform
    # )

    dataset, subsets = get_mesh_dataset(
        name=ckpt['config'].data_name,
        path=ckpt['config'].data_path,
        split_path=ckpt['config'].data_split,
        transform=transform
    )

    train_set, test_set = subsets['train'], subsets['val']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'],
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    )
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {ckpt_path}')

    # num_gpus = torch.cuda.device_count()
    # processes = []
    # for i in range(8):
    #     gpu_id = i % num_gpus
    #     p = multiprocessing.Process(target=sample_on_gpu, args=(
    #     i, gpu_id, test_set, num_samples, batch_size, num_steps, pos_only, center_pos_mode, sample_num_atoms,
    #     result_path, model))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()

    for i in range(10):
        data_id = i
        data = test_set[data_id]
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, num_samples,
            batch_size=batch_size, device=device,
            num_steps=num_steps,
            pos_only=pos_only,
            center_pos_mode=center_pos_mode,
            sample_num_atoms=sample_num_atoms
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list
        }
        logger.info('Sample done!')

        os.makedirs(result_path, exist_ok=True)
        torch.save(result, os.path.join(result_path, f'result_{data_id}.pt'))

    return result_path



if __name__ == '__main__':
    sample()

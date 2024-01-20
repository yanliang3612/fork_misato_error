import argparse
import os
import shutil
import cProfile
import pstats
import wandb

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset, get_mesh_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from pytorch_lightning import seed_everything
from sample_diffusion import sample
from evaluate_diffusion import evaluate
import torch.multiprocessing as mp



def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def parser_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--data_name', type=str, default="pl")
    parser.add_argument('--data_path', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--data_split', type=str, default='/mnt/workspace/dlly/targetdiff_mesh/data/icml_new_data_split.pt')
    parser.add_argument('--ligand_atom_mode', type=str, default='add_aromatic')
    parser.add_argument('--random_rot', type=bool, default=False)
    # model
    parser.add_argument('--loss_mesh_weight', type=float, default=1)
    parser.add_argument('--model_mean_type', type=str, default="C0")
    parser.add_argument('--beta_schedule', type=str, default="sigmoid")
    parser.add_argument('--beta_start', type=float, default=1.e-7)
    parser.add_argument('--beta_end', type=float, default=2.e-3)
    parser.add_argument('--v_beta_schedule', type=str, default='cosine')
    parser.add_argument('--v_beta_s', type=float, default=0.01)
    parser.add_argument('--num_diffusion_timesteps', type=int, default=1000)
    parser.add_argument('--loss_v_weight', type=float, default=100.)
    parser.add_argument('--sample_time_method', type=str, default='symmetric')

    parser.add_argument('--time_emb_dim', type=int, default=0)
    parser.add_argument('--time_emb_mode', type=str, default='simple')
    parser.add_argument('--center_pos_mode', type=str, default='protein')

    parser.add_argument('--node_indicator', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='uni_o2')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=9)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--edge_feat_dim', type=int, default=4)
    parser.add_argument('--num_r_gaussian', type=int, default=20)
    parser.add_argument('--knn', type=int, default=32)
    parser.add_argument('--num_node_types', type=int, default=8)

    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--cutoff_mode', type=str, default='knn')
    parser.add_argument('--ew_net_type', type=str, default='global')
    parser.add_argument('--num_x2h', type=int, default=1)
    parser.add_argument('--num_h2x', type=int, default=1)
    parser.add_argument('--r_max', type=float, default=10.)
    parser.add_argument('--x2h_out_fc', type=bool, default=False)
    parser.add_argument('--sync_twoup', type=bool, default=False)

    # train
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--n_acc_batch', type=int, default=2)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--pos_noise_std', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=8.0)
    parser.add_argument('--bond_loss_weight', type=float, default=1.0)

    # optimizer
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=5.e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--beta1', type=float, default=0.95)
    parser.add_argument('--beta2', type=float, default=0.999)

    # scheduler
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--factor', type=int, default=0.6)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_lr', type=int, default=1.e-6)

    # more
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default="test")

    return parser.parse_args()




def parser_args_sweep():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--data_name', type=str, default="pl")
    parser.add_argument('--data_path', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--data_split', type=str, default='/mnt/workspace/dlly/targetdiff_mesh/data/icml_new_data_split.pt')
    parser.add_argument('--ligand_atom_mode', type=str, default='add_aromatic')
    parser.add_argument('--random_rot', type=bool, default=False)


    # model
    parser.add_argument('--loss_mesh_weight', type=float, default=1)
    parser.add_argument('--model_mean_type', type=str, default="C0")
    parser.add_argument('--beta_schedule', type=str, default="sigmoid")
    parser.add_argument('--beta_start', type=float, default=1.e-7)
    parser.add_argument('--beta_end', type=float, default=2.e-3)
    parser.add_argument('--v_beta_schedule', type=str, default='cosine')
    parser.add_argument('--v_beta_s', type=float, default=0.01)
    parser.add_argument('--num_diffusion_timesteps', type=int, default=1000)
    parser.add_argument('--loss_v_weight', type=float, default=100.)
    parser.add_argument('--sample_time_method', type=str, default='symmetric')

    parser.add_argument('--time_emb_dim', type=int, default=0)
    parser.add_argument('--time_emb_mode', type=str, default='simple')
    parser.add_argument('--center_pos_mode', type=str, default='protein')

    parser.add_argument('--node_indicator', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='uni_o2')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=9)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--edge_feat_dim', type=int, default=4)
    parser.add_argument('--num_r_gaussian', type=int, default=20)
    parser.add_argument('--knn', type=int, default=32)
    parser.add_argument('--num_node_types', type=int, default=8)

    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--cutoff_mode', type=str, default='knn')
    parser.add_argument('--ew_net_type', type=str, default='global')
    parser.add_argument('--num_x2h', type=int, default=1)
    parser.add_argument('--num_h2x', type=int, default=1)
    parser.add_argument('--r_max', type=float, default=10.)
    parser.add_argument('--x2h_out_fc', type=bool, default=False)
    parser.add_argument('--sync_twoup', type=bool, default=False)

    # train
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--n_acc_batch', type=int, default=2)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--pos_noise_std', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=8.0)
    parser.add_argument('--bond_loss_weight', type=float, default=1.0)

    # optimizer
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=5.e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--beta1', type=float, default=0.95)
    parser.add_argument('--beta2', type=float, default=0.999)

    # scheduler
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--factor', type=int, default=0.6)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_lr', type=int, default=1.e-6)

    # more
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default="test")
    parser.add_argument("--sweep_id", type=str, default='')

    return parser.parse_known_args()[0]





def main_process(ckpt_path, result_path, num_processes):
    mp.set_start_method('spawn', force=True)

    processes = []
    for data_id in range(num_processes):
        p = mp.Process(target=sample, args=(ckpt_path, data_id, result_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



def main():
    args = parser_args()

    config_name = 'training'
    seed_everything(args.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)

    ckpt_dir = os.path.join("./logs_diffusion", f'{args.exp_name}_checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(args.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if args.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')

    dataset, subsets = get_mesh_dataset(
        name= args.data_name,
        path= args.data_path,
        split_path=args.data_split,
        transform=transform
    )

    # split_train_list = []
    # split_val_list = []
    # train_set_list = torch.load('./data/train_data_file_sdf_list.pt')
    # val_set_list = torch.load('./data/val_data_file_sdf_list.pt')
    # for i in range(len(dataset)):
    #     if dataset[i].ligand_filename in train_set_list:
    #         split_train_list.append(i)
    #     else:
    #         split_val_list.append(i)
    # dic = {'train':split_train_list, 'val':split_val_list}
    # torch.save(dic, './data/icml_new_data_split.pt')

    train_set, val_set = subsets['train'], subsets['val']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    # data_file_sdf_list = []
    # for i in range(len(dataset)):
    #     data_file_sdf_list.append(dataset[i].ligand_filename)
    #     print(i)
    # torch.save(dataset, './data/data_file_sdf_list.pt')

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        args,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(args, model)
    scheduler = utils_train.get_scheduler(args, optimizer)

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(args.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            protein_noise = torch.randn_like(batch.protein_pos) * args.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                ligand_mesh_pos = batch.ligand_mesh_pos,
                ligand_mesh_v = torch.repeat_interleave(batch.ligand_atom_feature_full,int(batch.ligand_mesh_pos.shape[0]/batch.ligand_pos.shape[0])),
                batch_ligand=batch.ligand_element_batch,
                batch_ligand_mesh = torch.repeat_interleave(batch.ligand_element_batch,int(batch.ligand_mesh_pos.shape[0]/batch.ligand_pos.shape[0])),
                batch_ligand_element = batch.ligand_element,
            )
            loss, loss_pos, loss_v, loss_mesh = results['loss'], results['loss_pos'], results['loss_v'], results['loss_mesh']
            loss = loss / args.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        logger.info(
            '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | mesh %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                it, loss, loss_pos, loss_v, loss_mesh, optimizer.param_groups[0]['lr'], orig_grad_norm
            )
        )
        if it % args.train_report_iter == 0:
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v,sum_loss_mesh, sum_n = 0, 0, 0, 0, 0
        sum_loss_bond, sum_loss_non_bond = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)

                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        ligand_mesh_pos=batch.ligand_mesh_pos,
                        ligand_mesh_v=torch.repeat_interleave(batch.ligand_atom_feature_full, int(
                            batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])),
                        batch_ligand=batch.ligand_element_batch,
                        batch_ligand_mesh=torch.repeat_interleave(batch.ligand_element_batch, int(
                            batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])),
                        batch_ligand_element=batch.ligand_element,
                    )
                    loss, loss_pos, loss_v, loss_mesh = results['loss'], results['loss_pos'], results['loss_v'], \
                                                        results['loss_mesh']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_mesh += float(loss_mesh) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=args.ligand_atom_mode)

        if args.scheduler_type == 'plateau':
            scheduler.step(avg_loss)
        elif args.scheduler_type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 |Loss mesh %.6f | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, loss_mesh, atom_auroc
            )
        )
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        ckpt_it = []
        # for it in range(1, args.max_iters + 1):
        #     # with torch.autograd.detect_anomaly():
        #     train(it)
        #     if it % args.val_freq == 0 or it == args.max_iters:
        #         val_loss = validate(it)
        #         if best_loss is None or val_loss < best_loss:
        #             logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
        #             best_loss, best_iter = val_loss, it
        #             ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
        #             ckpt_it.append(it)
        #             torch.save({
        #                 'config': args,
        #                 'model': model.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'scheduler': scheduler.state_dict(),
        #                 'iteration': it,
        #             }, ckpt_path)
        #         else:
        #             logger.info(f'[Validate] Val loss is not improved. '
        #                         f'Best val loss: {best_loss:.6f} at iter {best_iter}')
        best_it = max(ckpt_it)
        # test_ckpt_path = os.path.join(ckpt_dir, '%d.pt' % best_it)

        test_ckpt_path = os.path.join(ckpt_dir, '%d.pt' % 200)
        result_path = os.path.join(ckpt_dir, f'{best_it}sample_outs')
        os.makedirs(result_path, exist_ok=False)
        num_processes = 8
        main_process(test_ckpt_path, result_path, num_processes)
        evaluate_results = evaluate(result_path)
        print(evaluate_results)

    except KeyboardInterrupt:
        logger.info('Terminating...')




def sweep_fun():
    with wandb.init(project='mesh_sbdd', settings=wandb.Settings(start_method='fork')) as run:
        run.config.setdefaults(default_config)

        args = run.config

        config_name = 'training'
        seed_everything(args.seed)

        # Logging
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)

        ckpt_dir = os.path.join("./logs_diffusion", f'{args.exp_name}_checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        vis_dir = os.path.join(log_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        logger = misc.get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        logger.info(args)

        # Transforms
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(args.ligand_atom_mode)
        transform_list = [
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ]
        if args.random_rot:
            transform_list.append(trans.RandomRotation())
        transform = Compose(transform_list)

        # Datasets and loaders
        logger.info('Loading dataset...')

        dataset, subsets = get_mesh_dataset(
            name= args.data_name,
            path= args.data_path,
            split_path=args.data_split,
            transform=transform
        )

        # split_train_list = []
        # split_val_list = []
        # train_set_list = torch.load('./data/train_data_file_sdf_list.pt')
        # val_set_list = torch.load('./data/val_data_file_sdf_list.pt')
        # for i in range(len(dataset)):
        #     if dataset[i].ligand_filename in train_set_list:
        #         split_train_list.append(i)
        #     else:
        #         split_val_list.append(i)
        # dic = {'train':split_train_list, 'val':split_val_list}
        # torch.save(dic, './data/icml_new_data_split.pt')

        train_set, val_set = subsets['train'], subsets['val']
        logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

        # data_file_sdf_list = []
        # for i in range(len(dataset)):
        #     data_file_sdf_list.append(dataset[i].ligand_filename)
        #     print(i)
        # torch.save(dataset, './data/data_file_sdf_list.pt')

        # follow_batch = ['protein_element', 'ligand_element']
        collate_exclude_keys = ['ligand_nbh_list']
        train_iterator = utils_train.inf_iterator(DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys
        ))
        val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                                follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

        # Model
        logger.info('Building model...')
        model = ScorePosNet3D(
            args,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        # print(model)
        print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
        logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

        # Optimizer and scheduler
        optimizer = utils_train.get_optimizer(args, model)
        scheduler = utils_train.get_scheduler(args, optimizer)

        def train(it):
            model.train()
            optimizer.zero_grad()
            for _ in range(args.n_acc_batch):
                batch = next(train_iterator).to(args.device)

                protein_noise = torch.randn_like(batch.protein_pos) * args.pos_noise_std
                gt_protein_pos = batch.protein_pos + protein_noise
                results = model.get_diffusion_loss(
                    protein_pos=gt_protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,

                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    ligand_mesh_pos = batch.ligand_mesh_pos,
                    ligand_mesh_v = torch.repeat_interleave(batch.ligand_atom_feature_full,int(batch.ligand_mesh_pos.shape[0]/batch.ligand_pos.shape[0])),
                    batch_ligand=batch.ligand_element_batch,
                    batch_ligand_mesh = torch.repeat_interleave(batch.ligand_element_batch,int(batch.ligand_mesh_pos.shape[0]/batch.ligand_pos.shape[0])),
                    batch_ligand_element = batch.ligand_element,
                )
                loss, loss_pos, loss_v, loss_mesh = results['loss'], results['loss_pos'], results['loss_v'], results['loss_mesh']
                loss = loss / args.n_acc_batch
                loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | mesh %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, loss_mesh, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            if it % args.train_report_iter == 0:
                for k, v in results.items():
                    if torch.is_tensor(v) and v.squeeze().ndim == 0:
                        writer.add_scalar(f'train/{k}', v, it)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
                writer.add_scalar('train/grad', orig_grad_norm, it)
                writer.flush()


        def validate(it):
            # fix time steps
            sum_loss, sum_loss_pos, sum_loss_v,sum_loss_mesh, sum_n = 0, 0, 0, 0, 0
            sum_loss_bond, sum_loss_non_bond = 0, 0
            all_pred_v, all_true_v = [], []
            all_pred_bond_type, all_gt_bond_type = [], []
            with torch.no_grad():
                model.eval()
                for batch in tqdm(val_loader, desc='Validate'):
                    batch = batch.to(args.device)
                    batch_size = batch.num_graphs
                    t_loss, t_loss_pos, t_loss_v = [], [], []
                    for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                        time_step = torch.tensor([t] * batch_size).to(args.device)

                        results = model.get_diffusion_loss(
                            protein_pos=batch.protein_pos,
                            protein_v=batch.protein_atom_feature.float(),
                            batch_protein=batch.protein_element_batch,

                            ligand_pos=batch.ligand_pos,
                            ligand_v=batch.ligand_atom_feature_full,
                            ligand_mesh_pos=batch.ligand_mesh_pos,
                            ligand_mesh_v=torch.repeat_interleave(batch.ligand_atom_feature_full, int(
                                batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])),
                            batch_ligand=batch.ligand_element_batch,
                            batch_ligand_mesh=torch.repeat_interleave(batch.ligand_element_batch, int(
                                batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])),
                            batch_ligand_element=batch.ligand_element,
                        )
                        loss, loss_pos, loss_v, loss_mesh = results['loss'], results['loss_pos'], results['loss_v'], \
                                                            results['loss_mesh']

                        sum_loss += float(loss) * batch_size
                        sum_loss_pos += float(loss_pos) * batch_size
                        sum_loss_v += float(loss_v) * batch_size
                        sum_loss_mesh += float(loss_mesh) * batch_size
                        sum_n += batch_size
                        all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                        all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

            avg_loss = sum_loss / sum_n
            avg_loss_pos = sum_loss_pos / sum_n
            avg_loss_v = sum_loss_v / sum_n
            atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                                   feat_mode=args.ligand_atom_mode)

            if args.scheduler_type == 'plateau':
                scheduler.step(avg_loss)
            elif args.scheduler_type == 'warmup_plateau':
                scheduler.step_ReduceLROnPlateau(avg_loss)
            else:
                scheduler.step()

            logger.info(
                '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 |Loss mesh %.6f | Avg atom auroc %.6f' % (
                    it, avg_loss, avg_loss_pos, avg_loss_v * 1000, loss_mesh, atom_auroc
                )
            )
            writer.add_scalar('val/loss', avg_loss, it)
            writer.add_scalar('val/loss_pos', avg_loss_pos, it)
            writer.add_scalar('val/loss_v', avg_loss_v, it)
            writer.flush()
            return avg_loss

        try:
            best_loss, best_iter = None, None
            ckpt_it = []
            # for it in range(1, args.max_iters + 1):
            #     # with torch.autograd.detect_anomaly():
            #     train(it)
            #     if it % args.val_freq == 0 or it == args.max_iters:
            #         val_loss = validate(it)
            #         if best_loss is None or val_loss < best_loss:
            #             logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
            #             best_loss, best_iter = val_loss, it
            #             ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
            #             ckpt_it.append(it)
            #             torch.save({
            #                 'config': args,
            #                 'model': model.state_dict(),
            #                 'optimizer': optimizer.state_dict(),
            #                 'scheduler': scheduler.state_dict(),
            #                 'iteration': it,
            #             }, ckpt_path)
            #         else:
            #             logger.info(f'[Validate] Val loss is not improved. '
            #                         f'Best val loss: {best_loss:.6f} at iter {best_iter}')
            best_it = max(ckpt_it)
            # test_ckpt_path = os.path.join(ckpt_dir, '%d.pt' % best_it)

            test_ckpt_path = os.path.join(ckpt_dir, '%d.pt' % 200)
            result_path = os.path.join(ckpt_dir, f'{best_it}sample_outs')
            os.makedirs(result_path, exist_ok=False)
            num_processes = 8
            main_process(test_ckpt_path, result_path, num_processes)
            evaluate_results = evaluate(result_path)
            wandb.log(evaluate_results)

        except KeyboardInterrupt:
            logger.info('Terminating...')

# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    default_config = parser_args_sweep()
    wandb.agent(sweep_id=default_config.sweep_id, function=sweep_fun)
    sweep_fun()





# cProfile.run('main()', 'profile_stats')
# with open('./profile_report.txt', 'w') as f:
#     p = pstats.Stats('profile_stats', stream=f)
#     p.sort_stats('cumulative').print_stats(10)

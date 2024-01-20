import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.data import PDBProtein, parse_sdf_file, parse_sdf_npz_file
from datasets.pl_data import ProteinLigandData, torchify_dict


def find_file_and_check_existence(root_folder, filename):
    for root, dirs, files in os.walk(root_folder):
        if filename in files:
            return True, os.path.join(root, filename)
    return False, None

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, num, th,transform=None, version='final'):
        super().__init__()
        self.num = num
        self.th = th


        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}_test.lmdb')
        self.transform = transform
        self.db = None

        self.keys = None

        # if not os.path.exists(self.processed_path):
        #     print(f'{self.processed_path} does not exist, begin processing data')
        #     self._process()

        # if not os.path.exists(self.processed_path):
        print(f'{self.processed_path} does not exist, begin processing data')
        self._process()


    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                npz_filename = (ligand_fn.replace(".sdf", "_msms_mesh.npz")).replace('/', '_')
                if not os.path.exists(os.path.join("/mnt/workspace/dlly/targetdiff_mesh/data/crossdocked_pocket10_mesh6", npz_filename)):continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    """
                    mesh_file_add
                    """
                    npz_file = os.path.join("/mnt/workspace/dlly/targetdiff_mesh/data/crossdocked_pocket10_mesh6",npz_filename)
                    # ligand_dict = parse_sdf_npz_file(os.path.join(data_prefix, ligand_fn), npz_file, self.num, self.th)
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data


if __name__ == '__main__':
    import argparse
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/mnt/workspace/dlly/targetdiff_mesh/data/crossdocked_v1.1_rmsd1.0_pocket10")
    parser.add_argument('--num', type=int,default=5)
    parser.add_argument('--th', type=str, default=0.05)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path, args.num, args.th)
    print(len(dataset), dataset[0])
    # import os
    # import shutil
    #
    #
    # def copy_npz_files(src_directory, dest_directory):
    #     npz_count = 0  # 初始化计数器
    #
    #     # 确保目标目录存在
    #     if not os.path.exists(dest_directory):
    #         os.makedirs(dest_directory)
    #
    #     # 遍历源目录及其所有子目录
    #     for root, dirs, files in os.walk(src_directory):
    #         for file in files:
    #             if file.endswith('.npz'):
    #                 # 构建原始文件的完整路径
    #                 file_path = os.path.join(root, file)
    #
    #                 # 构建目标文件的完整路径
    #                 dest_file_path = os.path.join(dest_directory, file)
    #
    #                 # 防止同名文件覆盖，如果目标文件夹中已经有相同名称的文件，重命名新文件
    #                 if os.path.exists(dest_file_path):
    #                     name, ext = os.path.splitext(file)
    #                     dest_file_path = os.path.join(dest_directory, f"{name}_copy{ext}")
    #
    #                 # 复制文件
    #                 shutil.copy(file_path, dest_file_path)
    #
    #                 # 增加计数
    #                 npz_count += 1
    #
    #     return npz_count
    #
    #
    # # 指定源目录和目标目录
    # src_directory = '/mnt/workspace/dlly/targetdiff_mesh/data/crossdocked_pocket10_mesh4'  # 替换成你的源目录路径
    # dest_directory = '/mnt/workspace/dlly/targetdiff_mesh/data/crossdocked_pocket10_mesh6'  # 替换成你的目标目录路径
    #
    # # 执行函数并打印结果
    # npz_file_count = copy_npz_files(src_directory, dest_directory)
    # print(f"Total number of NPZ files copied: {npz_file_count}")







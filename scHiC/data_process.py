import os
import re
import numpy as np
from util.io import compactM, divide, pooling
from sklearn.model_selection import train_test_split

def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    data = np.genfromtxt(file_path, delimiter='\t')
    print()
    matrix_size = int(max(data[:, 0].max(), data[:, 1].max())) + 1
    contact_matrix = np.zeros((matrix_size, matrix_size))

    for row in data:
        row_idx = int(row[0])
        col_idx = int(row[1])
        contact_matrix[row_idx, col_idx] = row[2]

    return contact_matrix


def load_matrix2(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    # 使用 genfromtxt 从文件中加载数据，返回一个 numpy 数组
    matrix = np.genfromtxt(file_path, delimiter='\t')  # 根据实际情况调整分隔符
    return matrix

def load_singlecell_and_pseudobulk(cell_id, singlecell_dir, pseudobulk_dir):
    # 完整的文件路径，包括扩展名
    singlecell_file = os.path.join(singlecell_dir, f"{cell_id}_500000.matrix_chr2.txt")  # 单细胞文件名
    pseudobulk_file = os.path.join(pseudobulk_dir, f"pseudobulk_{cell_id}_500000.matrix_chr2.txt")  # 伪bulk文件名

    print(f"Loading singlecell matrix from {singlecell_file} and pseudobulk matrix from {pseudobulk_file}")

    singlecell_matrix = load_matrix(singlecell_file)
    pseudobulk_matrix = load_matrix2(pseudobulk_file)


    return singlecell_matrix, pseudobulk_matrix





def preprocess_data(singlecell_matrix, pseudobulk_matrix, chunk, stride, bound, lr_cutoff=100, hr_cutoff=255):
    singlecell_size = singlecell_matrix.shape[0]
    pseudobulk_size = pseudobulk_matrix.shape[0]

    # 找到更大的矩阵的尺寸
    max_size = max(singlecell_size, pseudobulk_size)

    # 对较小的矩阵进行零填充
    if singlecell_size < max_size:
        singlecell_matrix = np.pad(singlecell_matrix,
                                    ((0, max_size - singlecell_size), (0, max_size - singlecell_size)),
                                    mode='constant', constant_values=0)

    if pseudobulk_size < max_size:
        pseudobulk_matrix = np.pad(pseudobulk_matrix,
                                   ((0, max_size - pseudobulk_size), (0, max_size - pseudobulk_size)),
                                   mode='constant', constant_values=0)

    compact_idx = np.arange(max_size)

    print(f"singlecell_matrix shape: {singlecell_matrix.shape}")
    print(f"pseudobulk_matrix shape: {pseudobulk_matrix.shape}")
    print(f"compact_idx max: {np.max(compact_idx)}, compact_idx length: {len(compact_idx)}")

    singlecell_compact = compactM(singlecell_matrix, compact_idx)
    pseudobulk_compact = compactM(pseudobulk_matrix, compact_idx)

    singlecell_clamp = np.minimum(lr_cutoff, singlecell_compact)
    pseudobulk_clamp = np.minimum(hr_cutoff, pseudobulk_compact)

    singlecell_clamp /= lr_cutoff
    pseudobulk_clamp /= np.max(pseudobulk_clamp)

    div_singlecell, div_inds = divide(singlecell_clamp, chunk, stride, bound)
    div_pseudobulk, _ = divide(pseudobulk_clamp, chunk, stride, bound)

    div_singlecell_pool = pooling(div_singlecell, scale=1).numpy()

    print(f"Processed data for singlecell and pseudobulk.")
    return div_singlecell_pool, div_pseudobulk, div_inds


def save_processed_data(data, target, inds, output_dir, dataset_type):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_type}_dataset.npz")

    np.savez_compressed(output_file, data=data, target=target, inds=inds)
    print(f"Saved {dataset_type} dataset to {output_file}")

def process_all_cells(cell_ids, singlecell_dir, pseudobulk_dir, output_dir, chunk=40, stride=40, bound=201):
    print(f"Processing {len(cell_ids)} cell IDs...")

    all_singlecell_data = []
    all_pseudobulk_data = []
    all_inds = []

    for cell_id in cell_ids:
        try:
            singlecell_matrix, pseudobulk_matrix = load_singlecell_and_pseudobulk(cell_id, singlecell_dir, pseudobulk_dir)
            div_singlecell, div_pseudobulk, div_inds = preprocess_data(singlecell_matrix, pseudobulk_matrix, chunk, stride, bound)
            all_singlecell_data.append(div_singlecell)
            all_pseudobulk_data.append(div_pseudobulk)
            all_inds.append(div_inds)
        except Exception as e:
            print(f"Error processing cell ID {cell_id}: {e}")

    # Combine all data
    combined_singlecell_data = np.concatenate(all_singlecell_data)
    combined_pseudobulk_data = np.concatenate(all_pseudobulk_data)
    combined_inds = np.concatenate(all_inds)

    # Split the data into training, validation, and test sets
    train_data, temp_data, train_target, temp_target, train_inds, temp_inds = train_test_split(
        combined_singlecell_data, combined_pseudobulk_data, combined_inds, test_size=0.3, random_state=42)  # 70%训练集

    valid_data, test_data, valid_target, test_target, valid_inds, test_inds = train_test_split(
        temp_data, temp_target, temp_inds, test_size=0.5, random_state=42)  # 15%验证集和15%测试集

    # Save the datasets
    save_processed_data(train_data, train_target, train_inds, output_dir, 'H1Esc_train')
    save_processed_data(valid_data, valid_target, valid_inds, output_dir, 'H1Esc_valid')
    save_processed_data(test_data, test_target, test_inds, output_dir, 'H1Esc_test')


def get_cell_ids(singlecell_dir):
    cell_ids = []
    for filename in os.listdir(singlecell_dir):
        # 使用正则表达式匹配文件名中的细胞 ID
        match = re.search(r'(.+_human_\d+_[A-Z]+-[A-Z]+)_500000\.matrix_chr2\.txt$', filename)
        if match:
            cell_ids.append(match.group(1))  # 提取匹配的细胞 ID
    print(f"Found {len(cell_ids)} cell IDs: {cell_ids}")  # 打印提取到的ID
    return cell_ids



if __name__ == '__main__':
    singlecell_dir = '/home/graduates/Betsy/scHiC_data/chr/chr2/H1Esc/H1Esc.R2'
    pseudobulk_dir = '/home/graduates/Betsy/scHiC_data/pseudobulk/chr2/H1Esc/H1Esc.R2'
    output_dir = '/home/graduates/Betsy/scHiC_data/partition of dataset/chr2/H1Esc/H1Esc.R2'
    os.makedirs(pseudobulk_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    cell_ids = get_cell_ids(singlecell_dir)

    process_all_cells(cell_ids, singlecell_dir, pseudobulk_dir, output_dir)

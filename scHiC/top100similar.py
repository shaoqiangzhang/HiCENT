import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 定义你的文件夹路径
data_folder = "/home/graduates/Betsy/scHiC_data/chr/chr2/IMR90/IMR90-HAP1.R1"

# 获取文件夹中的所有文件
cell_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]

# 从文件名中提取细胞ID的函数
def extract_cell_id(filename):
    return filename.split('_')[2].split('.')[0]

# 加载接触矩阵的函数
def load_cell_matrix(file_path):
    data = np.genfromtxt(file_path, delimiter='\t', dtype=float, invalid_raise=False)

    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Data in file {file_path} is not in the expected format.")

    max_row_index = int(data[:, 0].max())
    max_col_index = int(data[:, 1].max())
    contact_matrix = np.zeros((max_row_index + 1, max_col_index + 1))

    for row in data:
        row_index = int(row[0])
        col_index = int(row[1])

        if row_index <= max_row_index and col_index <= max_col_index:
            contact_matrix[row_index, col_index] = row[2]
        else:
            print(f"Warning: Index out of bounds for row {row_index} and column {col_index}.")

    return contact_matrix

# 使用 cell_files 中的每个文件来加载接触矩阵，并生成 cell_matrices
cell_matrices = {}
for cell_file in cell_files:
    full_path = os.path.join(data_folder, cell_file)  # 拼接完整路径
    contact_matrix = load_cell_matrix(full_path)  # 加载接触矩阵
    cell_matrices[cell_file] = contact_matrix  # 使用文件名作为键
    print(f"Cell File: {cell_file}, Matrix shape: {contact_matrix.shape}, First few rows: {contact_matrix[:5]}")  # 打印信息

# 确定接触矩阵的统一形状
max_rows = max(matrix.shape[0] for matrix in cell_matrices.values())
max_cols = max(matrix.shape[1] for matrix in cell_matrices.values())

# 创建一个统一形状的矩阵列表，并进行填充
cell_matrix_list = np.zeros((len(cell_files), max_rows, max_cols))
for i, cell_file in enumerate(cell_files):
    matrix = cell_matrices[cell_file]
    cell_matrix_list[i, :matrix.shape[0], :matrix.shape[1]] = matrix  # 填充实际矩阵

# 计算余弦相似度时，直接使用接触矩阵而非均值
flattened_cell_matrices = np.array([matrix.flatten() for matrix in cell_matrix_list])

# 计算每个细胞接触矩阵的余弦相似度
similarity_matrix = cosine_similarity(flattened_cell_matrices)

# 提取 top 100 相关细胞
top_100_cells_dict = {}

for i, cell_file in enumerate(cell_files):
    top_100_indices = np.argsort(similarity_matrix[i, :])[-101:]  # -101 包括自己，-1 排除自己
    top_100_cells_dict[cell_file] = [cell_files[idx] for idx in top_100_indices]  # 使用文件名

# 保存结果
output_file = "/home/graduates/Betsy/scHiC_data/chr/chr2/IMR90/IMR90-HAP1.R1/top_100_similar_cells_cosine.txt"
with open(output_file, 'w') as f:
    for cell_file, top_100_cells in top_100_cells_dict.items():
        f.write(f"{cell_file}: {', '.join(top_100_cells)}\n")

print(f"Top 100 similar cells for each cell have been saved to {output_file}.")

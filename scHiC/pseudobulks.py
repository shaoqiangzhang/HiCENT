import os
import numpy as np

# 定义文件路径
data_folder = "/home/graduates/Betsy/scHiC_data/chr/chr2/GM12878/HFF-GM12878.R2"
similar_cells_file = os.path.join(data_folder, "top_100_similar_cells_cosine.txt")

# 读取每个细胞ID的前100个相似细胞
similar_cells_dict = {}
cell_file_dict = {}  # 用于存储细胞ID与文件名的对应关系

with open(similar_cells_file, "r") as f:
    for line in f:
        parts = line.strip().split(':')
        cell_id = parts[0].strip()
        similar_cells = parts[1].strip().split(', ')
        similar_cells_dict[cell_id] = similar_cells

        # 从文件名中提取细胞文件名并存储
        corresponding_file = f"human_{cell_id}.txt"  # 假设文件名为 human_{cell_id}.txt
        if corresponding_file in os.listdir(data_folder):
            cell_file_dict[cell_id] = corresponding_file

# 定义加载Hi-C矩阵的函数
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
        contact_matrix[row_index, col_index] = row[2]
    return contact_matrix

# 定义填充矩阵的函数（填充到相同大小）
def pad_matrix(contact_matrix, target_shape):
    rows, cols = contact_matrix.shape
    padded_matrix = np.zeros(target_shape)
    padded_matrix[:rows, :cols] = contact_matrix
    return padded_matrix

# 定义对称化函数
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# 生成伪bulk数据集
def generate_pseudobulk(cell_id, similar_cells, cell_files):
    pseudobulk_matrix = None
    max_rows, max_cols = 0, 0

    # 获取最大行列数
    for similar_cell in similar_cells:
        similar_files = [f for f in cell_files if f.startswith(f"{similar_cell}")]
        for cell_file in similar_files:
            full_path = os.path.join(data_folder, cell_file)
            contact_matrix = load_cell_matrix(full_path)
            max_rows = max(max_rows, contact_matrix.shape[0])
            max_cols = max(max_cols, contact_matrix.shape[1])

    # 初始化伪bulk矩阵
    pseudobulk_matrix = np.zeros((max_rows, max_cols))

    # 填充伪bulk矩阵
    for similar_cell in similar_cells:
        similar_files = [f for f in cell_files if f.startswith(f"{similar_cell}")]
        for cell_file in similar_files:
            full_path = os.path.join(data_folder, cell_file)
            contact_matrix = load_cell_matrix(full_path)
            contact_matrix_padded = pad_matrix(contact_matrix, (max_rows, max_cols))
            pseudobulk_matrix += contact_matrix_padded

    # 对称化处理
    pseudobulk_matrix = symmetrize(pseudobulk_matrix)
    print(f"pseudobulk_matrix shape before returning: {pseudobulk_matrix.shape}")
    return pseudobulk_matrix

# 获取文件夹中的所有单细胞文件
cell_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
output_folder = "/home/graduates/Betsy/scHiC_data/pseudobulk/chr2s/GM12878/HFF-GM12878.R2"
os.makedirs(output_folder, exist_ok=True)

# 生成每个细胞的伪bulk数据集
pseudobulk_results = {}
for cell_id, similar_cells in similar_cells_dict.items():
    pseudobulk_matrix = generate_pseudobulk(cell_id, similar_cells, cell_files)
    if pseudobulk_matrix is not None:
        original_file_name = cell_file_dict.get(cell_id, f"{cell_id}")  # 获取对应的原文件名
        output_file = os.path.join(output_folder, f"pseudobulk_{original_file_name}")  # 用原文件名命名
        print(f"Saving pseudobulk matrix for cell ID: {cell_id}, Shape: {pseudobulk_matrix.shape}")
        np.savetxt(output_file, pseudobulk_matrix, delimiter='\t')
        pseudobulk_results[cell_id] = pseudobulk_matrix
        print(f"Generated pseudobulk for cell ID: {cell_id}, Matrix shape: {pseudobulk_matrix.shape}")

print("All pseudobulk data saved.")

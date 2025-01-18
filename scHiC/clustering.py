import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 加载增强前后的数据
original_data_imr90 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/new/combined_original_data.npz')['data']
enhanced_data_imr90 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/new/combined_enhanced_data.npz')['data']

# 将数据展平成2D
original_data_imr90_reshaped = original_data_imr90.reshape(original_data_imr90.shape[0], -1)
enhanced_data_imr90_reshaped = enhanced_data_imr90.reshape(enhanced_data_imr90.shape[0], -1)

# 标准化数据
scaler = StandardScaler()
original_data_scaled = scaler.fit_transform(original_data_imr90_reshaped)
enhanced_data_scaled = scaler.fit_transform(enhanced_data_imr90_reshaped)

# 使用PCA降维
pca = PCA(n_components=2)
original_pca = pca.fit_transform(original_data_scaled)
enhanced_pca = pca.fit_transform(enhanced_data_scaled)

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
original_tsne = tsne.fit_transform(original_data_scaled)
enhanced_tsne = tsne.fit_transform(enhanced_data_scaled)

# 颜色设置
colors = ['b', 'g', 'r', 'c', 'm']  # 为五个细胞系分配不同的颜色

# 假设每个细胞系的数据已分配为相同的格式
cell_types = ['CellType1', 'CellType2', 'CellType3', 'CellType4', 'CellType5']  # 你的细胞系名称
# 假设每个数据集的数量相同，按细胞系索引获取颜色
original_indices = np.repeat(np.arange(len(cell_types)), original_data_imr90.shape[1])
enhanced_indices = np.repeat(np.arange(len(cell_types)), enhanced_data_imr90.shape[1])

# 绘制PCA结果并保存
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i, color in enumerate(colors):
    plt.scatter(original_pca[original_indices == i, 0], original_pca[original_indices == i, 1], c=color, label=cell_types[i])
plt.title('Original Data PCA')
plt.legend()
plt.savefig('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/original_data_pca.svg')  # 保存图像

plt.subplot(1, 2, 2)
for i, color in enumerate(colors):
    plt.scatter(enhanced_pca[enhanced_indices == i, 0], enhanced_pca[enhanced_indices == i, 1], c=color, label=cell_types[i])
plt.title('Enhanced Data PCA')
plt.legend()
plt.savefig('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/enhanced_data_pca.svg')  # 保存图像

plt.show()

# 绘制t-SNE结果并保存
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i, color in enumerate(colors):
    plt.scatter(original_tsne[original_indices == i, 0], original_tsne[original_indices == i, 1], c=color, label=cell_types[i])
plt.title('Original Data t-SNE')
plt.legend()
plt.savefig('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/original_data_tsne.svg')  # 保存图像

plt.subplot(1, 2, 2)
for i, color in enumerate(colors):
    plt.scatter(enhanced_tsne[enhanced_indices == i, 0], enhanced_tsne[enhanced_indices == i, 1], c=color, label=cell_types[i])
plt.title('Enhanced Data t-SNE')
plt.legend()
plt.savefig('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/enhanced_data_tsne.svg')  # 保存图像

plt.show()

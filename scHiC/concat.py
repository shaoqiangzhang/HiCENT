import numpy as np

# 加载五个原始数据集
original_data_1 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/GM12878/HFF-GM12878/GM12878_test_dataset.npz')['data']
original_data_2 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/H1Esc/H1Esc/H1Esc_test_dataset.npz')['data']
original_data_3 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/HAP1/HAP1_test_dataset.npz')['data']
original_data_4 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/HFF/H1Esc-HFF/HFF_test_dataset.npz')['data']
original_data_5 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/IMR90/IMR90-HAP1/IMR90_test_dataset.npz')['data']

# 加载五个增强后的数据集
enhanced_data_1 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/GM12878/HFF-GM12878/sr_output_test_dataset.npz')['data']
enhanced_data_2 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/H1Esc/H1Esc/sr_output_test_dataset.npz')['data']
enhanced_data_3 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/HAP1/sr_output_test_dataset.npz')['data']
enhanced_data_4 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/HFF/H1Esc-HFF/sr_output_test_dataset.npz')['data']
enhanced_data_5 = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/IMR90/IMR90-HAP1/sr_output_test_dataset.npz')['data']

# 合并五个原始数据集
combined_original_data = np.concatenate([original_data_1, original_data_2, original_data_3, original_data_4, original_data_5])

# 合并五个增强后的数据集
combined_enhanced_data = np.concatenate([enhanced_data_1, enhanced_data_2, enhanced_data_3, enhanced_data_4, enhanced_data_5])

# 保存合并后的原始数据集
np.savez_compressed('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/new/combined_original_data.npz', data=combined_original_data)

# 保存合并后的增强数据集
np.savez_compressed('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/new/combined_enhanced_data.npz', data=combined_enhanced_data)

# 验证加载合并后的数据
loaded_original_data = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/new/combined_original_data.npz')['data']
loaded_enhanced_data = np.load('/home/graduates/Betsy/scHiC_data/partition of dataset/chr1/compare/new/combined_enhanced_data.npz')['data']

print("Combined original data shape:", loaded_original_data.shape)
print("Combined enhanced data shape:", loaded_enhanced_data.shape)

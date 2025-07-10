from models.der import DER
import numpy as np
import logging
from torch.utils.data import DataLoader

class SimilarityBasedDER(DER):
    def __init__(self, args):
        super().__init__(args)
        # 可以添加新的参数来控制样本选择的策略
        self.min_samples = args.get("min_samples", 10)  # 每个类别最少的样本数
        self.max_samples = args.get("max_samples", 30)  # 每个类别最多的样本数
        self.similarity_threshold = args.get("similarity_threshold", 0.5)  # 相似度阈值

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars with similarity-based strategy...")
        for class_idx in range(self._known_classes, self._total_classes):
            # 获取新类别的数据
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            new_class_mean = np.mean(vectors, axis=0)
            new_class_mean = new_class_mean / np.linalg.norm(new_class_mean)

            # 计算与旧类的相似度
            max_similarity = 0
            if self._known_classes > 0:
                similarities = []
                for old_class_idx in range(self._known_classes):
                    old_class_mean = self._class_means[old_class_idx]
                    similarity = np.dot(new_class_mean, old_class_mean)
                    similarities.append(similarity)
                max_similarity = max(similarities)

            # 根据相似度动态调整样本数量
            if max_similarity > self.similarity_threshold:
                # 相似度高，选择更多样本
                adjusted_m = min(
                    self.max_samples,
                    int(m * (1 + max_similarity))
                )
            else:
                # 相似度低，选择较少样本
                adjusted_m = max(
                    self.min_samples,
                    int(m * (1 - max_similarity))
                )

            # 选择样本
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, adjusted_m + 1):
                S = np.sum(exemplar_vectors, axis=0) if exemplar_vectors else 0
                mu_p = (vectors + S) / k
                i = np.argmin(np.sqrt(np.sum((new_class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))
                exemplar_vectors.append(np.array(vectors[i]))
                vectors = np.delete(vectors, i, axis=0)
                data = np.delete(data, i, axis=0)

            # 保存选择的样本
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(adjusted_m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # 更新类别均值
            self._class_means[class_idx] = new_class_mean
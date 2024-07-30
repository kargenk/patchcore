import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class KCenterGreedy:
    """Greedy法による特徴量選択
    選択した点から最も遠いものから順に選択対象としていく
    """

    def __init__(self, X, y, seed, metric='euclidean'):
        self.X = X
        self.y = y
        self.name = 'kcenter'
        self.features = self._flatten_X()
        self.metric = metric
        self.min_distances = None
        self.already_selected = []

    def _flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    def select_samples(self, model, already_selected, n):
        # 特徴量の次元削減
        self.features = model.transform(self.X)

        # N個の特徴を選択する
        sample_indices = []
        for _ in tqdm(range(n)):
            if self.already_selected is None:
                # 初期化: Xの特徴数の中からランダムに一つ選択
                idx = np.random.choice(np.arange(self.X.shape[0]))
            else:
                # 現在選択されている特徴量との距離が最大の点（特徴量）を選択
                idx = np.argmax(self.min_distances)
            assert idx not in already_selected
            sample_indices.append(idx)

            # 選択した特徴量から他の全特徴量への距離を計算
            selected = self.features[[idx]]
            dist = pairwise_distances(self.features, selected, metric=self.metric)

            # クラスタの中心からの最小距離を更新
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                # 以前計算された距離と現時点で計算した距離の小さい方を残す
                self.min_distances = np.minimum(self.min_distances, dist)
        print(f'Maximum distance from cluster centers is {max(self.min_distances)}')

        self.already_selected = already_selected

        return sample_indices


# class GreedyCoresetSampler:
#     def __init__(self,)

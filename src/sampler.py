import numpy as np
from sklearn.metrics import pairwise_distances


class KCenterGreedy():
    def __init__(self, X, y, seed, metric='euclidean'):
        self.X = X
        self.y = y
        self.name = 'kcenter'
        self.features = self.flatten_X()
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    def select_batch(self, model, already_selected, N):
        self.features = model.transform(self.X)

        # N個の特徴を選択する
        new_batch = []
        for _ in range(N):
            if self.already_selected is None:
                # 初期化: Xの特徴数の中からランダムに一つ選択
                idx = np.random.choice(np.arange(self.n_obs))
            else:
                # 現在選択されている特徴量との距離が最大の点（特徴量）を選択
                idx = np.argmax(self.min_distances)
            assert idx not in already_selected

            # 特徴を選択して距離を計算
            x = self.features[[idx]]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            # 各特徴に対する最も小さな距離を残す
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                # 以前計算された距離と現時点で計算した距離の小さい方を残す
                self.min_distances = np.minimum(self.min_distances, dist)
            new_batch.append(idx)
        print(f'Maximum distance from cluster centers is {max(self.min_distances)}')

        self.already_selected = already_selected

        return new_batch
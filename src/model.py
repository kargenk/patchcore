import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

from plot_utils import inv_transform, save_anomaly_map
from sampler import KCenterGreedy

ROOT_DIR = Path(__file__).parents[1]
MODEL_DIR = ROOT_DIR.joinpath('models')
OUTPUT_DIR = ROOT_DIR.joinpath('outputs', 'back')


class PatchCore(nn.Module):
    def __init__(self, device='cpu', file_name: str = 'coreset_patch_features.pickle', out_dir: Path = OUTPUT_DIR):
        super().__init__()
        self.device = device
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        self.extractor = create_feature_extractor(self.model, {'layer2': 'mid1', 'layer3': 'mid2'}).to(device)
        self.memory_bank = []
        self.save_path = MODEL_DIR.joinpath(file_name)
        self.out_dir = out_dir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴抽出を行う

        Args:
            x (torch.Tensor): 入力

        Returns:
            dict[str, torch.Tensor]: 中間特徴量の辞書
                ex.) ResNet50 - layer2: [N, 512, H/8, W/8], layer3: [N, 1024, H/16, W/16]
        """
        self.feature = self.extractor(x)  # layer2: [N, 512, H/8, W/8], layer3: [N, 1024, H/16, W/16]
        return self.feature

    def add_memory(self, img_tensor) -> None:
        """位置毎の特徴量をMemoryBankに格納(バッチごとに処理を行う)

        Args:
            img_tensor (object): 入力画像[N, C, H, W]
        """
        patch_features = self.patchify(img_tensor)
        self.memory_bank.extend(patch_features.cpu().detach().numpy())  # [N * H * W] * C

    def save_memories(self, save_ratio: float = 0.01) -> None:
        """高速化のため、Random Projectionを使用して次元削減後、Greedy法を用いて特徴量のサンプリングを行って保存する.

        Args:
            save_ratio (float, optional): coresetとして何割の特徴量を保存するか.
                論文ではMVTecADデータセットにおいては0.01(1%)でも十分な精度. Defaults to 0.01.
        """
        patch_features = np.array(self.memory_bank)
        # 高速化のため、Random Projectionで特徴量の次元を削減, 'auto' => Johnson-Lindenstrauss lemma
        # 高次元のユークリッド空間内の要素をそれぞれの要素間の距離をある程度保ったまま、
        # 別の(低次元の)ユークリッド空間へ線型写像で移せるという補題
        # ref. paper 3.2
        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.random_projector.fit(patch_features)

        # Greedy法を用いて、特徴量の数をN個選択する
        selector = KCenterGreedy(patch_features, 0, 0)
        selected_idx = selector.select_samples(
            model=self.random_projector, already_selected=[], n=int(patch_features.shape[0] * save_ratio)
        )
        self.memory_bank_coreset = patch_features[selected_idx]
        print('full memory bank size : ', patch_features.shape)  # (245760, 1536)
        print('coreset memory bank size : ', self.memory_bank_coreset.shape)  # (245, 1536)

        # 特徴量の保存
        with self.save_path.open('wb') as f:
            pickle.dump(self.memory_bank_coreset, f)

    def inference(self, img_tensor, batch_idx):
        # 正常画像の特徴量を読み込む
        with self.save_path.open('rb') as f:
            self.memory_bank_coreset = pickle.load(f)  # M, Size=[num_coreset, feature_dim]
            memory_bank_coreset = torch.from_numpy(self.memory_bank_coreset).to(self.device)
        # テスト画像の特徴量を計算
        patch_features = self.patchify(img_tensor)  # P(x^test), Size=[N*H*W=1024, feature_dim=1536]

        # memo: dist[0]はpatch0からcoreset全てに対する距離を表す
        dist = torch.cdist(patch_features, memory_bank_coreset)  # [patch, coreset]
        min_dist, min_idx = torch.min(dist, dim=1)  # テストパッチ毎にcoresetの中で最も距離が近いものを取得
        s_idx = torch.argmax(min_dist)
        s_star = torch.max(min_dist)

        # ref. paper eq(6)
        m_test = patch_features[s_idx].unsqueeze(0)  # 異常パッチ（メモリバンク内の正常パッチ特徴から最も距離が遠いもの）
        m_star = memory_bank_coreset[min_idx[s_idx]].unsqueeze(0)  # メモリバンク内の最近傍のパッチ特徴
        weight_dist = torch.cdist(m_star, memory_bank_coreset)
        _, nn_idx = torch.topk(weight_dist, k=3, largest=False)

        # ref. paper eq(7)
        m_star_knn = torch.linalg.norm(m_test - memory_bank_coreset[nn_idx[0, 1:]], dim=1)
        # Transformerで用いられているテクニックを使用
        D = torch.sqrt(torch.tensor(patch_features.shape[1]))
        weight = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        img_level_score = weight * s_star

        # セグメンテーション画像の作成
        seg_map = min_dist.reshape(32, 32)
        seg_map_resized = cv2.resize(seg_map.cpu().detach().numpy(), (254, 254))
        seg_map_resized_blur = gaussian_filter(seg_map_resized, sigma=4)

        # 画像の保存
        imagenet_inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255]
        imagenet_inv_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.255]
        img_origin = inv_transform(img_tensor, imagenet_inv_mean, imagenet_inv_std)
        input_x = cv2.cvtColor(img_origin.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        file_name = f'{batch_idx :05d}'
        save_anomaly_map(self.out_dir, seg_map_resized_blur, input_x, file_name)

    @torch.inference_mode()
    def patchify(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """パッチ特徴量を作成する.

        Args:
            img_tensor (torch.Tensor): 入力画像のテンソル[N, C, H, W]

        Returns:
            torch.Tensor: 入力画像のパッチ特徴量[N*H*W, C]
        """
        features = self.extractor(img_tensor)
        # 位置に敏感にならないようにAverage Poolingで周囲と混ぜる（ぼかす）
        # ref. paper eq(2) and 4.4.1
        embeddings = []
        m = nn.AvgPool2d(3, 1, 1)
        for _, feature in features.items():
            embeddings.append(m(feature))

        def _emb_concat(embeddings: list[torch.Tensor]) -> torch.Tensor:
            """2つの中間特徴量のうち、サイズの小さい方をバイリニア補間でアップサンプリングしてからチャネル方向に結合.

            Args:
                embeddings (dict[str, torch.Tensor]): 浅くも深くもない位置の中間特徴量二つ

            Returns:
                torch.Tensor: チャネル方向に結合した中間特徴量
            """
            h1 = embeddings[0].size()[2]  # torch.Size([N, 512, H/8, W/8])
            h2 = embeddings[1].size()[2]  # torch.Size([N, 1024, H/16, W/16])
            s = int(h1 / h2)  # 2
            mid2 = F.interpolate(embeddings[1], scale_factor=s, mode='bilinear')
            return torch.cat([embeddings[0], mid2], dim=1)

        def _reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
            """N x H x W 個のチャネル方向特徴量（パッチ特徴量）のリストを作成.

            Args:
                embedding (torch.Tensor): torch.Size([N, 1536, H/8, W/8])

            Returns:
                torch.Tensor: torch.Size([1536])の特徴量がNHW個並んだテンソル[num_patch, filters]
            """
            # [N, C, H, W] => [N, H, W, C] => [N * H * W, C]
            patch_features = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])
            return patch_features

        patch_features = _emb_concat(embeddings)
        patch_features = _reshape_embedding(patch_features)

        return patch_features


if __name__ == '__main__':
    dummy = torch.randn((8, 3, 256, 256))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patchcore = PatchCore(device=device)
    patchcore.add_memory(dummy.to(device))
    patchcore.save_memories()

import cv2
import clip
import json
import torch
import umap
import hydra
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Union, Tuple, List, Dict


class ClusterImages:
    def __init__(self):
        self.max_images = 3500
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self._get_logger()

    def get_best_k(self, images_embedding: np.ndarray, kmax: int = 10) -> int:
        wss_scores = self.calculate_WSS(images_embedding, kmax)
        sil_scores = self.calculate_silhouette_score(images_embedding, kmax)
        best_k = self.plot_k_choosing(sil_scores, wss_scores, kmax)
        return best_k

    @staticmethod
    def plot_k_choosing(sil_scores: List[float], wss_scores: List[int], kmax: int = 10) -> int:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        ax.plot(range(1, kmax + 1), wss_scores, color="blue", marker="o", label="WSS")
        ax.set_xlabel("Number of clusters", fontsize=16)
        ax.set_yticks([])
        ax.set_xticks(range(1, kmax + 1))
        ax.set_ylim(0, np.max(wss_scores) + 1000)

        ax2 = ax.twinx()
        ax2.plot(range(2, kmax + 1), sil_scores, color="green", marker="x", label="Silhouette score")
        ax2.vlines(
            x=np.argmax(sil_scores) + 2,
            ymin=0.0,
            ymax=np.max(sil_scores),
            color="red",
            linestyles="dashed",
            label="Best k",
        )
        ax2.set_yticks([])

        fig.suptitle("Elbow Method vs Silhouette Method", fontsize=20)
        fig.legend(loc="upper right", ncol=3)
        fig.tight_layout()

        # plt.show()

        return np.argmax(sil_scores) + 2

    @staticmethod
    def calculate_silhouette_score(points: np.ndarray, kmax: int = 10) -> List[float]:
        sil = []
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            labels = kmeans.labels_
            sil.append(silhouette_score(points, labels, metric="euclidean"))
        return sil

    @staticmethod
    def calculate_WSS(points: np.ndarray, kmax: int = 10) -> List[int]:
        sse = []
        for k in range(1, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0

            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

            sse.append(curr_sse)
        return sse

    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def write_words_clusters_to_json(possible_words: Dict[int, List[str]], json_path: Union[str, Path]):
        json_data = {str(i): possible_words[i] for i in possible_words.keys()}
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

    def calc_top_descriptors(
        self, descriptors: List[str], preprocessed_images: np.ndarray, kmeans_labels: List[np.array]
    ) -> Dict[int, List[str]]:
        tokenized_labels = clip.tokenize(descriptors).to(self.device)
        labels_count_for_class = {
            class_id: {label: 0 for label in descriptors} for class_id in np.unique(kmeans_labels)
        }
        for class_id, image in tqdm(
            zip(kmeans_labels, preprocessed_images),
            desc="finding best descriptors for each cluster",
            total=preprocessed_images.__len__(),
        ):
            logits = self.model(image, tokenized_labels)[0]
            top_k_labels = [descriptors[i] for i in logits.topk(5).indices[0]]
            for label in top_k_labels:
                labels_count_for_class[class_id][label] += 1 / kmeans_labels[kmeans_labels == class_id].shape[0]

        df = pd.DataFrame(labels_count_for_class)
        max_vals = df.idxmax(1)
        possible_words = {i: max_vals[max_vals == i].index.tolist() for i in np.unique(kmeans_labels)}

        return possible_words

    @staticmethod
    def cluster_images_w_umap(images_embedding: np.ndarray, n_clusters: int) -> List[np.array]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(images_embedding)
        return kmeans.labels_

    def get_encoded_images(self, images_dir: Union[Path, str]) -> Tuple[List[np.array], List[np.array], List[np.array]]:

        images = []
        encoded_images = []
        preprocessed_images = []
        images_generator = [file for file in Path(images_dir).rglob("*.png") if "side" not in file.stem]

        for idx, image in enumerate(tqdm(images_generator, total=len(images_generator), desc="Encoding images")):
            images.append(cv2.imread(image.as_posix()))
            image = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
            preprocessed_images.append(image)
            with torch.no_grad():
                image_features = self.model.encode_image(image).half().cpu().numpy()
            encoded_images.append(image_features)
            if idx == self.max_images:
                self.logger.warning(f"Reached max images: {self.max_images} - stopping encoding")
                break

        encoded_images = np.concatenate(encoded_images, axis=0)
        return encoded_images, preprocessed_images, images

    def cluster_images(
        self,
        images_dir: Union[Path, str],
        descriptors: List[str],
        out_path: Union[str, Path],
        json_name: str = "words_clusters",
    ) -> None:
        if isinstance(images_dir, Path):
            images_dir = images_dir.as_posix()
        if isinstance(out_path, str):
            out_path = Path(out_path)
        encoded_images, preprocessed_images, images = self.get_encoded_images(images_dir=images_dir)
        # images_embedding = umap.UMAP(n_neighbors=300, min_dist=0.0, metric="euclidean").fit_transform(encoded_images)
        best_k = self.get_best_k(encoded_images)
        kmeans_labels = self.cluster_images_w_umap(encoded_images, n_clusters=best_k)
        # delete unnecessary variables to free GPU memory
        del encoded_images
        # del images_embedding
        torch.cuda.empty_cache()
        possible_words = self.calc_top_descriptors(descriptors, preprocessed_images, kmeans_labels)
        self.write_words_clusters_to_json(possible_words, out_path / f"{json_name}.json")


@hydra.main(config_path="../../config", config_name="cluster_images")
def main(cfg: DictConfig) -> None:
    clusterer = ClusterImages()
    descriptors = list(cfg.descriptors)
    clusterer.cluster_images(
        images_dir=cfg.images_dir,
        descriptors=descriptors,
        out_path=cfg.out_path,
        json_name=cfg.images_dir.split("/")[-1],
    )


if __name__ == "__main__":
    main()

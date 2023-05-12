import json
import hydra
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import wordnet
from omegaconf import DictConfig
from itertools import permutations
from typing import Union, Dict, Any, Set, List
from clip2mesh.utils import Utils
from clip2mesh.choosing_descriptors_utils import ChoosingDescriptorsUtils


class ChoosingDescriptors(ChoosingDescriptorsUtils):
    def __init__(
        self,
        images_dir: Union[Path, str],
        max_num_of_descriptors: int,
        min_num_of_descriptors: int,
        descriptors_clusters_json: str,
        corr_threshold: float = 0.6,
        output_dir: Union[Path, str] = None,
        descriptors_to_keep: List[str] = None,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.utils = Utils()
        self.corr_threshold = corr_threshold
        self.images_dir: Path = Path(images_dir)
        self.max_num_of_descriptors = max_num_of_descriptors
        self.min_num_of_descriptors = min_num_of_descriptors
        self.descriptors_to_keep = descriptors_to_keep if descriptors_to_keep is not None else []
        if output_dir is not None:
            self.output_dir: Path = Path(output_dir)
        if descriptors_clusters_json is not None:
            self.clusters = self.get_clusters(Path(descriptors_clusters_json))
        self._get_logger()

    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger("choosing_descriptors")

    def _preprocess_2_words_descriptor(self, descriptor: str) -> str:
        splitted_to_words = descriptor.split(" ")
        if len(splitted_to_words) > 1:
            return True
        else:
            return False

    def _get_synonyms(self, descriptor: str) -> Set[str]:
        multiple_words_descriptor = self._preprocess_2_words_descriptor(descriptor)
        preprocessed_descriptor = descriptor.split(" ")[0] if multiple_words_descriptor else descriptor
        synonyms = []
        for syn in wordnet.synsets(preprocessed_descriptor):
            for l in syn.lemmas():
                synonyms.append(l.name())
        if multiple_words_descriptor:
            return set([f"{synonym} {descriptor.split(' ')[1]}" for synonym in synonyms])
        else:
            return set(synonyms)

    def _get_antonyms(self, descriptor: str) -> Set[str]:
        multiple_words_descriptor = self._preprocess_2_words_descriptor(descriptor)
        preprocessed_descriptor = descriptor.split(" ")[0] if multiple_words_descriptor else descriptor
        antonyms = []
        for syn in wordnet.synsets(preprocessed_descriptor):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        if multiple_words_descriptor:
            return set([f"{antonym} {descriptor.split(' ')[1]}" for antonym in antonyms])
        else:
            return set(antonyms)

    def get_dfs(self, jsons_dir: Union[Path, str]):
        json_files = list(Path(jsons_dir).rglob("*_labels.json"))
        df = pd.DataFrame()
        for json_file in tqdm(json_files, desc="Loading json files", total=len(json_files)):
            with open(json_file, "r") as f:
                json_data = json.load(f)
                df = pd.concat([df, pd.DataFrame(json_data)], axis=0)
        if "broad shoulders" in df.columns:
            df = df.drop("broad shoulders", axis=1)
        if "built" in df.columns:
            df = df.drop("built", axis=1)
        df = df.apply(lambda x: [y[0] for y in x])

        # get variances
        variances = df.var(axis=0)
        variances.sort_values(ascending=False, inplace=True)
        variances = pd.DataFrame(zip(variances.index, variances.values), columns=["descriptor", "variance"])

        # get corrlation matrix between descriptors
        corr_df = pd.DataFrame(columns=["descriptor_1", "descriptor_2", "correlation"])
        permut_list = []
        for perm in permutations(df.columns, 2):
            permut_list.append(perm)
        for perm in tqdm(permut_list, desc="Calculating correlations", total=len(permut_list)):
            corr_df = pd.concat(
                [
                    corr_df,
                    pd.DataFrame(
                        {
                            "descriptor_1": [perm[0]],
                            "descriptor_2": [perm[1]],
                            "correlation": [np.corrcoef(df[perm[0]], df[perm[1]])[0, 1]],
                        }
                    ),
                ],
                axis=0,
            )
        return corr_df, variances

    @staticmethod
    def get_descriptor_iou(descriptor: str, ious_df: pd.DataFrame) -> float:
        return ious_df[(ious_df["descriptor_1"] == descriptor) | (ious_df["descriptor_2"] == descriptor)]["iou"].mean()

    @staticmethod
    def get_clusters(descriptors_clusters_json: Path):
        with open(descriptors_clusters_json, "r") as f:
            clusters = json.load(f)
        return clusters

    def check_if_descriptor_is_correlated_with_chosen_descriptors(
        self, descriptor: str, chosen_descriptors: Dict[str, Dict[str, Any]], correlations_df: pd.DataFrame
    ) -> bool:
        for chosen_descriptor in chosen_descriptors:
            descriprtor_correlations = correlations_df[
                (
                    (correlations_df["descriptor_1"] == descriptor)
                    & (correlations_df["descriptor_2"] == chosen_descriptor)
                )
            ]
            if abs(descriprtor_correlations["correlation"][0]) > self.corr_threshold:
                return True, chosen_descriptor, descriprtor_correlations["correlation"][0]
        return False, None, None

    def initial_filter(self):
        # get the correlations and variances of the data
        correlations_df, variances = self.get_dfs(self.images_dir)
        if self.verbose:
            self.logger.info(f"There are total of {len(variances)} descriptors")

        # get the clusters
        if hasattr(self, "clusters"):
            chosen_descriptors = {cluster_id: {} for cluster_id in self.clusters}
            # add column of cluster to summary df
            variances["cluster"] = variances["descriptor"].apply(
                lambda x: self.find_cluster_of_descriptor(x, self.clusters)
            )
        else:
            chosen_descriptors = {0: {}}
            variances["cluster"] = np.zeros(len(variances)).astype(int)

        # if there are descriptors with None in "cluster" column, remove them
        if variances[variances["cluster"].isnull()].shape[0] > 0:
            if self.verbose:
                self.logger.info(
                    f"There are {variances[variances['cluster'].isnull()].shape[0]} descriptors with no cluster"
                )
            variances = variances[~variances["cluster"].isnull()]

        # if there are descriptors to keep, add them to the chosen descriptors
        if self.descriptors_to_keep is not None:
            if self.verbose:
                self.logger.info(f"Keeping {self.descriptors_to_keep}")
            for descriptor in self.descriptors_to_keep:
                cluster_id = self.find_cluster_of_descriptor(descriptor, self.clusters)
                descriptor_var = variances[variances["descriptor"] == descriptor]["variance"].values[0]
                chosen_descriptors[cluster_id][descriptor] = {"variance": descriptor_var}

        # start iterating over the clusters
        for cluster, cluster_df in variances.groupby("cluster"):

            if self.verbose:
                self.logger.info(f"Working on cluster {cluster}")

            # sort the cluster df by variance
            cluster_df = cluster_df.sort_values(by="variance", ascending=False)

            # iterate over the cluster descriptors
            for i, (_, row) in enumerate(cluster_df.iterrows()):

                # get the descriptor and its variance
                descriptor = row["descriptor"]
                descriptor_var = row["variance"]

                if descriptor in chosen_descriptors[cluster]:
                    if self.verbose:
                        self.logger.info(f"{descriptor} already chosen")
                    continue

                if self.verbose:
                    self.logger.info(f"{i} most variance descriptor - {descriptor}")

                # if this is the descriptor with the highest variance - add it
                if chosen_descriptors[cluster] == {}:

                    if self.verbose:
                        self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                    descriptor = row["descriptor"]

                    chosen_descriptors[cluster][descriptor] = {
                        "variance": descriptor_var,
                    }

                else:

                    # check if the descriptor is correlated with the chosen descriptors
                    corr_test = self.check_if_descriptor_is_correlated_with_chosen_descriptors(
                        descriptor, chosen_descriptors[cluster], correlations_df
                    )
                    if not corr_test[0]:

                        if self.verbose:
                            self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                        chosen_descriptors[cluster][descriptor] = {
                            "variance": descriptor_var,
                        }

                    else:

                        if self.verbose:
                            self.logger.info(
                                f"{row['descriptor']} is correlated with {corr_test[1]} with corr {corr_test[2]}"
                            )
            if self.verbose:
                self.logger.info(f"currently chosen descriptors: {chosen_descriptors[cluster]}")
                self.logger.info(f"Finished cluster {cluster}")
                print("*" * 100)

        if self.verbose:
            self.logger.info(
                f"There are {sum([len(descriptors) for descriptors in chosen_descriptors.values()])} chosen descriptors in initial filter"
            )
            self.logger.info(f"chosen descriptors: {[list(chosen_descriptors[x].keys()) for x in chosen_descriptors]}")
        return chosen_descriptors, correlations_df, variances

    def final_filter(self, chosen_descriptors: Dict[str, Dict[str, float]], correlations_df: pd.DataFrame):
        """
        After the initial filter, we want to remove descriptors that are correlated with each other
        or avoid synonyms and antonyms
        """
        # create df of chosen descriptors sorted by variance
        if self.verbose:
            print()
            self.logger.info("Final filter")
        chosen_descriptors_df = pd.DataFrame(self.flatten_dict_of_dicts(chosen_descriptors)).T.sort_values(
            by="variance", ascending=False
        )

        # create a subset of the correlations df with only the chosen descriptors
        subset_correlation_df = correlations_df[(correlations_df["descriptor_1"].isin(chosen_descriptors_df.index))]

        removed_descriptors = []
        chosed_descriptors = []

        # iterate over the chosen descriptors
        for idx, (descriptor, data) in enumerate(chosen_descriptors_df.iterrows()):

            # if the descriptor was removed in a previous iteration - skip it
            if descriptor in removed_descriptors:
                continue

            # if this is the first descriptor - add it, since it has the highest variance
            if idx == 0:
                chosed_descriptors.append(descriptor)
                continue

            # get the cluster of the current descriptor
            descriptor_cluster = self.find_cluster_of_descriptor(descriptor, chosen_descriptors)

            if self.verbose:
                self.logger.info(f"Descriptor {descriptor} has variance {data['variance']}")

            # the check is across all clusters except the current one, since it was already checked
            for cluster_id in chosen_descriptors.keys():

                # if this is the current cluster - skip it
                if cluster_id == descriptor_cluster:
                    continue

                # create a subset of the correlations df with only the chosen descriptors from the current cluster
                corr_df = subset_correlation_df[
                    (subset_correlation_df["descriptor_1"] == descriptor)
                    & subset_correlation_df["descriptor_2"].isin(chosen_descriptors[cluster_id])
                ]

                # create a subset of the highly correlated descriptors
                high_corr_df = corr_df[corr_df["correlation"] > self.corr_threshold]
                if high_corr_df.index.__len__() > 0:

                    # iterate over the highly correlated descriptors, and remove them
                    corr_descriptors = high_corr_df["descriptor_2"].values
                    for corr_descriptor, corr_value in zip(corr_descriptors, high_corr_df["correlation"].values):
                        if self.verbose:
                            self.logger.info(
                                f"Descriptor {descriptor} is correlated with {corr_descriptor} with correlation {corr_value}"
                            )
                        if corr_descriptor in chosed_descriptors and descriptor not in self.descriptors_to_keep:
                            if self.verbose:
                                self.logger.info(f"Removing {descriptor} because {corr_descriptor} is already chosen")
                            removed_descriptors.append(descriptor)
                            break

                        elif (
                            corr_descriptor not in removed_descriptors
                            and corr_descriptor not in self.descriptors_to_keep
                        ):
                            if self.verbose:
                                self.logger.info(
                                    f"Removing {corr_descriptor} because it is correlated with {descriptor}"
                                )
                            removed_descriptors.append(corr_descriptor)

                # get synonyms of the current descriptor, and check if they are in the chosen descriptors - if so - remove them
                synonyms = self._get_synonyms(descriptor)
                if synonyms is not None:
                    for synonym in synonyms:
                        if synonym in chosen_descriptors:
                            if descriptor in self.descriptors_to_keep:
                                if self.verbose:
                                    self.logger.info(
                                        f"Removing {synonym} because it is a synonym of {descriptor} and {descriptor} is in the descriptors to keep"
                                    )
                                removed_descriptors.append(synonym)
                            else:
                                if self.verbose:
                                    self.logger.info(f"Removing {descriptor} because it is a synonym of {synonym}")
                                removed_descriptors.append(descriptor)

                # get antonyms of the current descriptor, and check if they are in the chosen descriptors - if so - remove them
                antonyms = self._get_antonyms(descriptor)
                if antonyms is not None:
                    for antonym in antonyms:
                        if antonym in chosed_descriptors:
                            if descriptor in self.descriptors_to_keep:
                                if self.verbose:
                                    self.logger.info(
                                        f"Removing {antonym} because it is an antonym of {descriptor} and {descriptor} is in the descriptors to keep"
                                    )
                                removed_descriptors.append(antonym)
                            else:
                                if self.verbose:
                                    self.logger.info(f"Removing {descriptor} because it is an antonym of {antonym}")
                                removed_descriptors.append(descriptor)

                # if the descriptor passed all the checks - add it to the chosen descriptors
                if descriptor not in chosed_descriptors and descriptor not in removed_descriptors:
                    chosed_descriptors.append(descriptor)

        # create a df of the removed descriptors
        removed_descriptors_df = chosen_descriptors_df[chosen_descriptors_df.index.isin(removed_descriptors)]

        # remove the removed descriptors from the chosen descriptors df
        chosen_descriptors_df = chosen_descriptors_df.drop(removed_descriptors)
        return chosen_descriptors_df, removed_descriptors_df

    def choose(self) -> Dict[str, Dict[str, float]]:
        """
        choose the descriptors, and return a dict of the chosen descriptors per cluster
        """
        # initial filter - remove descriptors with low variance, and high correlation
        chosen_descriptors, correlations_df, variance_df = self.initial_filter()

        # final filter - remove descriptors that are correlated across clusters, synonyms and antonyms
        final_filtered_chosen_descriptors, removed_descriptors_df = self.final_filter(
            chosen_descriptors, correlations_df
        )

        # create a dict of the chosen descriptors per cluster
        final_choose = {cluster_id: {} for cluster_id in self.clusters}
        for desc in final_filtered_chosen_descriptors.index:
            final_choose[self.find_cluster_of_descriptor(desc, chosen_descriptors)][
                desc
            ] = final_filtered_chosen_descriptors.loc[desc]["variance"]

        # check if the number of chosen descriptors is within the allowed range
        number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)

        # if the number of chosen descriptors is too high - reduce the number of descriptors
        if number_of_descriptors > self.max_num_of_descriptors:

            if self.verbose:
                self.logger.info(f"Too many descriptors ({number_of_descriptors}) - reducing")

            while number_of_descriptors > self.max_num_of_descriptors:
                final_choose = self.reduce_descriptor(final_choose)
                number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)

        # if the number of chosen descriptors is too low - increase the number of descriptors
        elif number_of_descriptors < self.min_num_of_descriptors:

            if self.verbose:
                self.logger.info(f"Too few descriptors ({number_of_descriptors}) - increasing")

            iterator = 0

            while number_of_descriptors < self.min_num_of_descriptors:

                skip = False

                # start with the discriptors that were removed in the final filter
                if iterator < removed_descriptors_df.shape[0]:

                    # get the cluster of the descriptor
                    cluster_id = self.find_cluster_of_descriptor(
                        removed_descriptors_df.iloc[iterator].name, self.clusters
                    )

                    synonyms = self._get_synonyms(removed_descriptors_df.iloc[iterator].name)
                    antonyms = self._get_antonyms(removed_descriptors_df.iloc[iterator].name)

                    # check if the descriptor is a synonym or antonym of a chosen descriptor - if so - skip it
                    if synonyms is not None:
                        for synonym in synonyms:
                            if synonym in self.flatten_dict_of_dicts(final_choose):
                                if self.verbose:
                                    self.logger.info(
                                        f"Descriptor {removed_descriptors_df.iloc[iterator].name} is a synonym of {synonym} - skipping"
                                    )
                                skip = True
                    if antonyms is not None:
                        for antonym in antonyms:
                            if antonym in self.flatten_dict_of_dicts(final_choose):
                                if self.verbose:
                                    self.logger.info(
                                        f"Descriptor {removed_descriptors_df.iloc[iterator].name} is an antonym of {antonym} - skipping"
                                    )
                                skip = True
                    if not skip:
                        (final_choose[cluster_id]).update(
                            {removed_descriptors_df.iloc[iterator].name: removed_descriptors_df.iloc[iterator].variance}
                        )
                    number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)
                    iterator += 1

                # when all the descriptors that were removed in the final filter were added - start adding descriptors with high variance
                else:
                    for _, row in variance_df.iterrows():
                        skip = False
                        if row.descriptor not in self.flatten_dict_of_dicts(final_choose):
                            cluster_id = self.find_cluster_of_descriptor(row.descriptor, self.clusters)

                            synonyms = self._get_synonyms(row.descriptor)
                            antonyms = self._get_antonyms(row.descriptor)

                            # check if the descriptor is a synonym or antonym of a chosen descriptor - if so - skip it
                            if synonyms is not None:
                                for synonym in synonyms:
                                    if synonym in self.flatten_dict_of_dicts(final_choose):
                                        if self.verbose:
                                            self.logger.info(
                                                f"Descriptor {row.descriptor} is a synonym of {synonym} - skipping"
                                            )
                                        skip = True
                            if antonyms is not None:
                                for antonym in antonyms:
                                    if antonym in self.flatten_dict_of_dicts(final_choose):
                                        if self.verbose:
                                            self.logger.info(
                                                f"Descriptor {row.descriptor} is an antonym of {antonym} - skipping"
                                            )
                                        skip = True
                            if not skip:
                                final_choose[cluster_id][row.descriptor] = row.variance

                            number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)
                            if number_of_descriptors >= self.min_num_of_descriptors:
                                break
                    break

        if hasattr(self, "output_dir"):
            with open(self.output_dir / f"chosen_descriptors.json", "w") as f:
                json.dump(final_choose, f)

        return final_choose


@hydra.main(config_path="../../config", config_name="choose_algorithm")
def main(cfg: DictConfig) -> None:
    choosing_descriptors = ChoosingDescriptors(**cfg)
    final_choose = choosing_descriptors.choose()
    print(f"Chosen descriptors: {final_choose}")
    print("number of chosen descriptors: ", choosing_descriptors.get_num_of_chosen_descriptors(final_choose))


if __name__ == "__main__":
    main()

import json
import logging
import pandas as pd
from typing import Tuple, Dict, Union, List


class ChoosingDescriptorsUtils:
    def __init__(self, verbose: bool = False):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

    def choose_between_2_descriptors(
        self, df: pd.DataFrame, first_descriptor: str, second_descriptor: str
    ) -> Tuple[str, float]:
        first_descriptor_avg_iou = df[
            (df["descriptor_1"] == first_descriptor) | (df["descriptor_2"] == first_descriptor)
        ]["iou"].mean()
        second_descriptor_avg_iou = df[
            (df["descriptor_1"] == second_descriptor) | (df["descriptor_2"] == second_descriptor)
        ]["iou"].mean()
        if self.verbose:
            self.logger.info(f"{first_descriptor} iou: {first_descriptor_avg_iou}")
        if self.verbose:
            self.logger.info(f"{second_descriptor} iou: {second_descriptor_avg_iou}")
        if first_descriptor_avg_iou < second_descriptor_avg_iou:
            if self.verbose:
                self.logger.info(f"chose {first_descriptor} with iou {first_descriptor_avg_iou}")
            return first_descriptor, first_descriptor_avg_iou
        else:
            if self.verbose:
                self.logger.info(f"chose {second_descriptor} with iou {second_descriptor_avg_iou}")
            return second_descriptor, second_descriptor_avg_iou

    @staticmethod
    def flatten_dict_of_dicts(dict_of_dicts: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        flattened_dict = {}
        for value in dict_of_dicts.values():
            flattened_dict.update(value)
        return flattened_dict

    @staticmethod
    def get_number_of_unique_descriptors(df: Union[pd.DataFrame, str]) -> int:
        if isinstance(df, str):
            df = pd.read_csv(df)
        unique_descriptors = set(df["descriptor_1"].unique().tolist() + df["descriptor_2"].unique().tolist())
        return len(unique_descriptors)

    def initial_filter(
        self,
        df_path: str,
        descriptors_groups_json: str,
        max_descriptors_per_cluster: int = 6,
        min_descriptors_overall: int = 1,
    ) -> Dict[int, Dict[str, float]]:

        # initializations
        df = pd.read_csv(df_path)
        descriptors_groups = json.load(open(descriptors_groups_json, "r"))
        total_descriptors = self.get_number_of_unique_descriptors(df)
        finalists_descriptors = {}
        all_possible_descriptors = {}

        # create a dictionary of all possible descriptors and their iou in case this function will return a lower number of descriptors than the minimum
        for cluster, descriptors in descriptors_groups.items():
            all_possible_descriptors[cluster] = {}
            for descriptor in descriptors:
                descriptor_iou = df[(df["descriptor_1"] == descriptor) | (df["descriptor_2"] == descriptor)][
                    "iou"
                ].mean()
                all_possible_descriptors[cluster][descriptor] = descriptor_iou

        # check if there are enough descriptors
        if total_descriptors < min_descriptors_overall:
            if self.verbose:
                self.logger.info(f"Total number of descriptors is smaller than {min_descriptors_overall}")
            return None, all_possible_descriptors

        # print some info
        if self.verbose:
            self.logger.info(f"Total number of descriptors to choose from: {total_descriptors}")
            self.logger.info(f"Total number of clusters: {len(descriptors_groups)}")

        # iterate over clusters
        for cluster, descriptors in descriptors_groups.items():

            # check if there are enough descriptors in the cluster
            # if not, return the single descriptor
            if len(descriptors) == 1:
                descriptor_iou = df[
                    (df["descriptor_1"] == second_descriptor) | (df["descriptor_2"] == second_descriptor)
                ]["iou"].mean()
                finalists_descriptors[cluster] = {descriptors[0]: descriptor_iou}

            # if there are enough descriptors in the cluster
            else:

                # filter the dataframe to only include the descriptors in the cluster
                group_df = df[df["descriptor_1"].isin(descriptors) & df["descriptor_2"].isin(descriptors)]
                group_df = group_df.sort_values("iou", ascending=False)
                final_candidates = {}

                # iterate over the groups
                # for group_idx in group_df["group"].unique():

                chosed_descriptors = {}

                # iterate over the rows in the group
                # for _, row in group_df[group_df["group"] == group_idx].iterrows():
                for _, row in group_df.iterrows():

                    if self.verbose:
                        print(f"*" * 50)
                    if self.verbose:
                        self.logger.info(f"choosing between: {row['descriptor_1']} | {row['descriptor_2']}")

                    # define the descriptors
                    first_descriptor = row["descriptor_1"]
                    second_descriptor = row["descriptor_2"]

                    # choose one of the descriptors
                    chosen_descriptor, chosen_descriptor_iou = self.choose_between_2_descriptors(
                        group_df, first_descriptor, second_descriptor
                    )

                    # if the chosen descriptor is not in the chosed_descriptors dict, add it
                    if chosen_descriptor not in chosed_descriptors.keys():

                        # if the dict is empty, add the descriptor
                        if chosed_descriptors == {}:
                            if self.verbose:
                                self.logger.info(f"first descriptor chosen -> {chosen_descriptor}")
                            chosed_descriptors[chosen_descriptor] = chosen_descriptor_iou

                        # if the dict is not empty, check if the chosen descriptor is better than the others
                        else:

                            if self.verbose:
                                self.logger.info(f"iterating over chosed descriptors -> {chosed_descriptors}")

                            # iterate over the chosed descriptors
                            add_descriptor = False
                            for descriptor in chosed_descriptors.keys():

                                if self.verbose:
                                    self.logger.info(f"choosing between: {descriptor} | {chosen_descriptor}")

                                # choose one of the descriptors
                                sub_chosen_descriptor, _ = self.choose_between_2_descriptors(
                                    group_df, descriptor, chosen_descriptor
                                )

                                # if the chosen descriptor is not the same as the current one, it means that the current descriptor is worse, hence break
                                if sub_chosen_descriptor != chosen_descriptor:
                                    if self.verbose:
                                        self.logger.info(
                                            f"{chosen_descriptor} has higher iou than {descriptor}, hence {descriptor} is chosen"
                                        )
                                    break

                                # if the chosen descriptor is not the same as the current one, it means that the current descriptor is better, hence add the chosen descriptor
                                else:
                                    add_descriptor = True

                            # if the chosen descriptor is better than all the others, add it
                            if add_descriptor:

                                if self.verbose:
                                    self.logger.info(
                                        f"{chosen_descriptor} has lower iou than all chosed descriptors, hence {chosen_descriptor} is chosen"
                                    )

                                chosed_descriptors[chosen_descriptor] = chosen_descriptor_iou

                    if self.verbose:
                        self.logger.info(chosed_descriptors)

                # add the chosen descriptors to the final candidates
                for descriptor, iou in chosed_descriptors.items():
                    if descriptor not in final_candidates.keys():
                        final_candidates[descriptor] = iou

                if self.verbose:
                    print(f"*" * 50)

                # sort the final candidates by iou
                possible_options = df["descriptor_1"].unique().tolist()
                possible_options = possible_options + [
                    item for item in df["descriptor_2"].unique() if item not in possible_options
                ]

                if self.verbose:
                    self.logger.info(f"possible_options: {possible_options} | total {len(possible_options)}")
                if self.verbose:
                    self.logger.info(f"final candidates: {final_candidates} | total {len(final_candidates)}")

                sorted_descriptors_by_iou_ascending = sorted(final_candidates, key=final_candidates.get, reverse=False)[
                    :max_descriptors_per_cluster
                ]
                finalists_descriptors[cluster] = {
                    descriptor_name: final_candidates[descriptor_name]
                    for descriptor_name in sorted_descriptors_by_iou_ascending
                }

        return finalists_descriptors, all_possible_descriptors

    def reduce_descriptor(self, dict_of_desctiptors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        # flatten the dict of dicts and sort it by variance
        flattened_dict = self.flatten_dict_of_dicts(dict_of_desctiptors)
        sorted_dict = sorted(flattened_dict, key=flattened_dict.get, reverse=False)

        # get the minimal variance descriptor, but verify that it is not in the descriptors to keep
        idx = 0
        keep_searching = True
        while keep_searching:
            minimal_var_descriptor = sorted_dict[idx]
            if minimal_var_descriptor not in self.descriptors_to_keep:
                keep_searching = False
            else:
                idx += 1
        cluster_of_descriptor = self.find_cluster_of_descriptor(minimal_var_descriptor, dict_of_desctiptors)

        if self.verbose:
            print(f"removing {minimal_var_descriptor} from cluster {cluster_of_descriptor}")
        del dict_of_desctiptors[cluster_of_descriptor][minimal_var_descriptor]

        clusters_to_delete = []
        for cluster in dict_of_desctiptors.keys():
            if dict_of_desctiptors[cluster] == {}:
                clusters_to_delete.append(cluster)
        for cluster in clusters_to_delete:
            del dict_of_desctiptors[cluster]
        return dict_of_desctiptors

    @staticmethod
    def find_cluster_of_descriptor(
        descriptor: str, dict_of_desctiptors: Dict[str, Union[Dict[str, float], List[str]]]
    ) -> int:
        for cluster, descriptors_dict in dict_of_desctiptors.items():
            if isinstance(descriptors_dict, dict):
                if descriptor in descriptors_dict.keys():
                    return cluster
            else:
                if descriptor in descriptors_dict:
                    return cluster

    @staticmethod
    def get_num_of_chosen_descriptors(dict_of_desctiptors: Dict[str, Dict[str, float]]) -> int:
        num_of_descriptors = 0
        for descriptors in dict_of_desctiptors.values():
            num_of_descriptors += len(descriptors)
        return num_of_descriptors

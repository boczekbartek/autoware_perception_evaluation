from logging import getLogger
import os
from typing import List
from typing import Tuple

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.objects_filter import filter_ground_truth_objects
from awml_evaluation.evaluation.matching.objects_filter import filter_tp_objects
from awml_evaluation.evaluation.object_result import DynamicObjectWithResult

logger = getLogger(__name__)


class MapConfig:
    """[summary]
    Config for mAP calculation

    Attributes:
        self.target_labels (List[AutowareLabel]): Target labels to evaluate
        self.matching_mode (MatchingMode): Matching mode like distance between the center of
                                           the object, 3d IoU
        self.matching_threshold (float): Threshold for matching the predicted object and
                                         ground truth
    """

    def __init__(
        self,
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> None:
        """[summary]

        Args:
            target_labels (List[AutowareLabel]): Target labels to evaluate
            matching_mode (MatchingMode): Matching mode like distance between the center of
                                          the object, 3d IoU
            matching_threshold (float): Threshold for matching the predicted object and
                                        ground truth
        """
        self.target_labels: List[AutowareLabel] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold: float = matching_threshold


class Map:
    """[summary]
    mAP class

    Attributes:
        self.map_config (MapConfig): The config for mAP calculation
        self.aps (List[Ap]): The list of AP (Average Precision) for each label
        self.map (float): mAP value
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects: List[DynamicObject],
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> None:
        """[summary]

        Args:
            object_results (DynamicObjectWithResult): The list of Object result
            object_results (List[DynamicObjectWithResult]): [description]
            ground_truth_objects (List[DynamicObject]): [description]
            target_labels (List[AutowareLabel]): [description]
            matching_mode (MatchingMode): [description]
            matching_threshold (float): [description]
        """
        if not target_labels:
            logger.warning(f"target_labels is empty ({target_labels})")
            return

        self.map_config = MapConfig(
            target_labels,
            matching_mode,
            matching_threshold,
        )

        self.aps: List[Ap] = []
        for target_label in self.map_config.target_labels:
            ap_ = Ap(
                object_results,
                ground_truth_objects,
                [target_label],
                matching_mode,
                matching_threshold,
            )
            self.aps.append(ap_)

        # calculate mAP
        sum_ap: float = 0.0
        for ap in self.aps:
            sum_ap += ap.ap
        self.map: float = sum_ap / len(target_labels)


class Ap:
    """[summary]
    AP class

    Attributes:
        self.target_labels (List[AutowareLabel]): Target labels to evaluate
        self.matching_mode (MatchingMode): Matching mode like distance between the center of
                                           the object, 3d IoU
        self.matching_threshold (float): Threshold for matching the predicted object and
                                         ground truth
        self.ground_truth_objects_num (int): The number of ground truth objects
        self.tp_list (List[int]): The list of the number of TP (True Positive) objects ordered
                                  by confidence
        self.fp_list (List[int]): The list of the number of FP (False Positive) objects ordered
                                  by confidence
        self.ap (float): AP (Average Precision)
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects: List[DynamicObject],
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> None:
        """[summary]

        Args:
            object_results (List[DynamicObjectWithResult]) : The results to each predicted object
            ground_truth_objects (List[DynamicObject]) : The ground truth objects for the frame
            target_labels (List[AutowareLabel]): Target labels to evaluate
            matching_mode (MatchingMode): Matching mode like distance between the center of
                                           the object, 3d IoU
            matching_threshold (float): Threshold for matching the predicted object and
        """

        self.target_labels: List[AutowareLabel] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold: float = matching_threshold

        # filter predicted object and results by iou_threshold and target_labels
        filtered_object_results: List[DynamicObjectWithResult] = filter_tp_objects(
            object_results=object_results,
            target_labels=self.target_labels,
            matching_mode=self.matching_mode,
            matching_threshold=self.matching_threshold,
        )
        # TODO sort confidence

        filtered_ground_truth_objects: List[DynamicObject] = filter_ground_truth_objects(
            objects=ground_truth_objects,
            target_labels=self.target_labels,
        )
        self.ground_truth_objects_num: int = len(filtered_ground_truth_objects)

        # tp and fp from object results ordered by confidence
        self.tp_list: List[int] = []
        self.fp_list: List[int] = []
        self.tp_list, self.fp_list = self._calculate_tp_fp(
            filtered_object_results, self.ground_truth_objects_num
        )

        # caliculate precision recall
        precision_list: List[float] = []
        recall_list: List[float] = []
        precision_list, recall_list = self.get_precision_recall_list()

        # AP
        self.ap: float = self._calculate_ap(precision_list, recall_list)

    def save_precision_recall_graph(
        self,
        result_directory: str,
        frame_name: str,
    ) -> None:
        """[summary]
        Save visualization image of precision and recall

        Args:
            result_directory (str): The directory path to save images
            frame_name (str): The frame name
        """

        base_name = f"{frame_name}_precision_recall_iou{self.iou_threshold}_"
        target_str = f"{_get_flat_str(self.target_labels)}"
        file_name = base_name + target_str + ".png"
        file_path = os.join(result_directory, file_name)

        precision_list: List[float] = []
        recall_list: List[float] = []
        precision_list, recall_list = self.get_precision_recall_list(
            self.tp_list,
            self.fp_list,
            self.ground_truth_objects_num,
        )

        # TODO impl save png
        # save(file_path, recall_list, precision_list)

    def get_precision_recall_list(
        self,
    ) -> Tuple[List[float], List[float]]:
        """[summary]
        Calculate precision recall

        Returns:
            Tuple[List[float], List[float]]: tp_list and fp_list

        Example:
            state
                self.tp_list = [1, 1, 2, 3]
                self.fp_list = [0, 1, 1, 1]
            return
                precision_list = [1.0, 0.5, 0.67, 0.75]
                recall_list = [0.25, 0.25, 0.5, 0.75]
        """
        precisions_list: List[float] = [0.0 for i in range(len(self.tp_list))]
        recalls_list: List[float] = [0.0 for i in range(len(self.tp_list))]

        for i in range(len(precisions_list)):
            precisions_list[i] = float(self.tp_list[i]) / (i + 1)
            if self.ground_truth_objects_num > 0:
                recalls_list[i] = float(self.tp_list[i]) / self.ground_truth_objects_num
            else:
                recalls_list[i] = 0.0

        return precisions_list, recalls_list

    @staticmethod
    def _calculate_tp_fp(
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects_num: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate TP (true positive) and FP (false positive)

        Args:
            object_results (List[DynamicObjectWithResult]): the list of objects with result
            ground_truth_objects_num (int): the number of ground truth objects

        Return:
            Tuple[tp_list, fp_list]

            tp_list (List[float]): the list of TP ordered by object confidence
            fp_list (List[float]): the list of FP ordered by object confidence

        Example:
            whether object label is correct [True, False, True, True]
            return
                tp_list = [1, 1, 2, 3]
                fp_list = [0, 1, 1, 1]
        """

        if len(object_results) == 0 and ground_truth_objects_num != 0:
            logger.warning("The size of object_results is 0")
            return [], []
        object_results_num = len(object_results)
        tp_list: List[int] = [0 for i in range(object_results_num)]
        fp_list: List[int] = [0 for i in range(object_results_num)]

        # init
        if object_results[0].is_label_correct:
            tp_list[0] = 1
            fp_list[0] = 0
        else:
            tp_list[0] = 0
            fp_list[0] = 1

        for i in range(1, len(object_results)):
            if object_results[i].is_label_correct:
                tp_list[i] = tp_list[i - 1] + 1
            else:
                tp_list[i] = tp_list[i - 1]
            fp_list[i] = i + 1 - tp_list[i]

        return tp_list, fp_list

    @staticmethod
    def _calculate_ap(
        precision_list: List[float],
        recall_list: List[float],
    ) -> float:
        """[summary]
        Calculate AP (average precision)

        Args:
            precision_list (List[float]): The list of precision
            recall_list (List[float]): The list of recall

        Returns:
            float: AP

        Example:
            precision_list = [1.0, 0.5, 0.67, 0.75]
            recall_list = [0.25, 0.25, 0.5, 0.75]

            max_precision_list: List[float] = [0.75, 1.0, 1.0]
            max_precision_recall_list: List[float] = [0.75, 0.25, 0.0]

            ap = 0.75 * (0.75 - 0.25) + 1.0 * (0.25 - 0.0)
               = 0.625

        """

        if len(precision_list) == 0:
            return 0.0

        max_precision_list: List[float] = [precision_list[-1]]
        max_precision_recall_list: List[float] = [recall_list[-1]]

        for i in reversed(range(len(recall_list) - 1)):
            if precision_list[i] > max_precision_list[-1]:
                max_precision_list.append(precision_list[i])
                max_precision_recall_list.append(recall_list[i])

        # append min recall
        max_precision_list.append(max_precision_list[-1])
        max_precision_recall_list.append(0.0)

        ap: float = 0.0
        for i in range(len(max_precision_list) - 1):
            score: float = max_precision_list[i] * (
                max_precision_recall_list[i] - max_precision_recall_list[i + 1]
            )
            ap += score

        return ap


def _get_flat_str(str_list: List[str]) -> str:
    """
    Example:
        a = _get_flat_str([aaa, bbb, ccc])
        print(a) # aaa_bbb_ccc
    """
    output = ""
    for one_str in str_list:
        output = f"{output}_{one_str}"
    return output
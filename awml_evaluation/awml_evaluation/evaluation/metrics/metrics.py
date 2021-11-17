from typing import List

from awml_evaluation.common.dataset import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.configure import MetricsScoreConfig
from awml_evaluation.evaluation.metrics.detection.map import Map
from awml_evaluation.evaluation.object_result import DynamicObjectWithResult


class MetricsScore:
    """[summary]

    Attributes:
        self.config (MetricsScoreConfig): The config for metrics calculation
        self.maps (List[Map]): The list of mAP class object. Each mAP is different from threshold
                               for matching (ex. IoU0.3)
    """

    def __init__(
        self,
        metrics_config: MetricsScoreConfig,
    ) -> None:
        """[summary]
        Args:
            metrics_config (MetricsScoreConfig) A config for metrics calculation
        """
        self.config: MetricsScoreConfig = metrics_config

        # for detection metrics
        self.maps: List[Map] = []

    def __str__(self) -> str:
        """[summary]
        Str method
        """
        str_: str = "\n"
        for map_ in self.maps:
            str_ += f"{map_.map_config.matching_mode.value} {map_.map_config.matching_threshold}"
            str_ += f" mAP: {map_.map}"

            object_num: int = 0
            for ap_ in map_.aps:
                object_num += ap_.ground_truth_objects_num
            str_ += f" (object num {object_num})"
            for ap_ in map_.aps:
                target_str: str = ""
                for target in ap_.target_labels:
                    target_str += target.value
                str_ += f", AP {target_str}: {ap_.ap}"
            str_ += "\n"
        return str_

    def evaluate(
        self,
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluate API

        Args:
            object_results (List[DynamicObjectWithResult]): The list of Object result
            ground_truth_objects (List[DynamicObject]): The ground truth objects
        """
        self._evaluation_detection(object_results, ground_truth_objects)
        self._evaluation_tracking(object_results)
        self._evaluation_prediction(object_results)

    def _evaluation_detection(
        self,
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Calculate detection metrics

        Args:
            object_results (List[DynamicObjectWithResult]): The list of Object result
            ground_truth_objects (List[DynamicObject]): The ground truth objects
        """
        if self.config.map_thresholds_center_distance:
            for distance_threshold_ in self.config.map_thresholds_center_distance:
                map_ = Map(
                    object_results,
                    ground_truth_objects,
                    self.config.target_labels,
                    MatchingMode.CENTERDISTANCE,
                    distance_threshold_,
                )
                self.maps.append(map_)
        if self.config.map_thresholds_iou:
            for iou_threshold_ in self.config.map_thresholds_iou:
                map_ = Map(
                    object_results,
                    ground_truth_objects,
                    self.config.target_labels,
                    MatchingMode.IOU3d,
                    iou_threshold_,
                )
                self.maps.append(map_)
        if self.config.map_thresholds_center_distance:
            for distance_threshold_ in self.config.map_thresholds_plane_distance:
                map_ = Map(
                    object_results,
                    ground_truth_objects,
                    self.config.target_labels,
                    MatchingMode.PLANEDISTANCE,
                    distance_threshold_,
                )
                self.maps.append(map_)

    def _evaluation_tracking(
        self,
        object_results: List[DynamicObjectWithResult],
    ) -> None:
        """[summary]
        Calculate tracking metrics

        Args:
            object_results (List[DynamicObjectWithResult]): The list of Object result
        """
        pass

    def _evaluation_prediction(
        self,
        object_results: List[DynamicObjectWithResult],
    ) -> None:
        """[summary]
        Calculate prediction metrics

        Args:
            object_results (List[DynamicObjectWithResult]): The list of Object result
        """
        pass
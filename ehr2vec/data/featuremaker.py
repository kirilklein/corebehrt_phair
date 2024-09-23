import pandas as pd
from typing import Tuple
from ehr2vec.data.creators import BaseCreator


class FeatureMaker:
    def __init__(self, config):
        self.config = config

        self.features = {
            "concept": [],
        }

        self.order = {"background": -1}
        self.creator_names = list(self.config.keys()) + ["death"]
        self.creators = {
            creator.id: creator
            for creator in BaseCreator.__subclasses__()
            if creator.id in self.creator_names
        }
        assert len(self.creators) == len(
            self.creator_names
        ), f"Found {len(self.creators)} creators but {len(self.creator_names)} were expected."
        self.pipeline = self.create_pipeline()

    def __call__(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> Tuple[dict, list]:
        """Insert features into the concepts dataframe. Inplace operation."""
        for creator in self.pipeline:
            concepts = creator(concepts, patients_info)
            concepts["CONCEPT"] = concepts["CONCEPT"].astype(str)
        features, pids = self.create_features(concepts)

        return features, pids

    def create_pipeline(self) -> list:
        """Create the pipeline of feature creators."""
        # Pipeline creation
        pipeline = []
        for id in self.creators:
            creator = self.creators[id](self.config)
            pipeline.append(creator)
            if getattr(creator, "feature", None) is not None:
                self.features[creator.feature] = []

        # Reordering
        pipeline_creators = [
            creator.id for creator in pipeline if hasattr(creator, "id")
        ]
        for id, pos in self.order.items():
            if id in pipeline_creators:
                creator = pipeline.pop(pipeline_creators.index(id))
                if pos == -1:
                    pos = len(pipeline)
                pipeline.insert(pos, creator)

        return pipeline

    def create_features(self, concepts: pd.DataFrame) -> Tuple[dict, list]:
        # Add standard info
        pids = []
        for pid, patient in concepts.groupby("PID", sort=False):
            pids.append(pid)
            for feature, value in self.features.items():
                value.append(patient[feature.upper()].tolist())

        return self.features, pids

import logging
import pandas as pd
from ehr2vec.data.utils import Utilities

logger = logging.getLogger(__name__)


class BaseCreator:
    """
    Base class for feature creators. Subclasses should implement the create method.
    The concepts dataframe starts out with the basic features and is passed through the pipeline of creators,
    iteratively adding new features.
    There is some dependency between the creators, e.g. the DeathCreator should be run at last.
    """

    def __init__(self, config: dict):
        self.config = config

    def __call__(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        return self.create(concepts, patients_info)

    def create(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def get_segment_column(concepts):
        if "ADMISSION_ID" in concepts.columns:
            return "ADMISSION_ID"
        elif "SEGMENT" in concepts.columns:
            return "SEGMENT"
        else:
            raise KeyError("No segment column found in concepts")

    @staticmethod
    def find_column(patients_info: pd.DataFrame, match: str):
        """Check if a column containing the match string exists in patients_info and return it."""
        columns = patients_info.columns.str.lower().str.contains(match.lower())
        if any(columns):
            return patients_info.columns[columns][0]
        raise KeyError(f'No column containing "{match}" found in patients_info')


class AgeCreator(BaseCreator):
    feature = id = "age"

    def create(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        birthdates = self._create_birthdate_map(patients_info)
        concepts["AGE"] = self._calculate_ages(concepts, birthdates)
        return concepts

    @staticmethod
    def _create_birthdate_map(patients_info: pd.DataFrame) -> dict:
        birthdate_col = BaseCreator.find_column(patients_info, "birth")
        logger.info(f"Creating age feature using birthdate column: {birthdate_col}")
        return pd.Series(
            patients_info[birthdate_col].values, index=patients_info["PID"]
        ).to_dict()

    def _calculate_ages(self, concepts: pd.DataFrame, birthdates: dict) -> pd.Series:
        ages = (
            concepts["TIMESTAMP"] - concepts["PID"].map(birthdates)
        ).dt.days / 365.25
        if self.config.age.get("round"):
            ages = ages.round(self.config.age.get("round"))
        return ages


class AbsposCreator(BaseCreator):
    """
    This creator is used to create the 'ABSPOS' feature.
    It calculates the absolute position of events from a given origin point.
    """

    feature = id = "abspos"

    def create(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        concepts["ABSPOS"] = Utilities.get_abspos_from_origin_point(
            concepts["TIMESTAMP"], self.config.abspos
        )
        return concepts


class SegmentCreator(BaseCreator):
    """
    Assigning visit numbers (segment) to each event based on admission ids or existing segment.
    """

    feature = id = "segment"

    def create(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        concepts["SEGMENT"] = self._assign_segments(concepts)
        return concepts

    def _assign_segments(self, concepts: pd.DataFrame) -> pd.Series:
        return concepts.groupby("PID")[self.get_segment_column(concepts)].transform(
            lambda x: pd.factorize(x)[0] + 1
        )


class BackgroundCreator(BaseCreator):
    """
    Assigning background features (gender, ethnicity...) to each patient. The corresponding abspos is set to birthdate.
    Age will be -1 and segment will be 0.
    """

    id = "background"
    prepend_token = "BG_"

    def create(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        background = {}
        background["PID"] = self._repeat_pids(patients_info)
        background["CONCEPT"] = self._create_background_concepts(patients_info)
        background["SEGMENT"] = 0
        background["AGE"] = -1
        background["ABSPOS"] = self._calculate_abspos_for_background(patients_info)

        background = pd.DataFrame(background)
        return pd.concat([background, concepts])

    def _repeat_pids(self, patients_info: pd.DataFrame) -> list:
        """Repeat each PID for each background feature."""
        return patients_info["PID"].tolist() * len(self.config.background)

    def _calculate_abspos_for_background(self, patients_info: pd.DataFrame) -> dict:
        """Abspos represents the timestamp of the birthdate for each patient."""
        abspos = Utilities.get_abspos_from_origin_point(
            patients_info[self.find_column(patients_info, "birth")], self.config.abspos
        )
        return abspos.to_list() * len(self.config.background)

    def _create_background_concepts(self, patients_info: pd.DataFrame) -> list:
        concepts = []
        for col in self.config.background:
            background_concepts = self._generate_concepts_for_column(patients_info, col)
            concepts.extend(background_concepts)
        return concepts

    def _generate_concepts_for_column(
        self, patients_info: pd.DataFrame, column: str
    ) -> list:
        """
        Generate background concepts for each value in the column.
        The concepts are created by concatenating the prepend_token (BG_) with column name and the value.
        """
        return [
            (self.prepend_token + column + "_" + str(value))
            for value in patients_info[column]
        ]


class DeathCreator(BaseCreator):
    """
    This creator is used to append death information to the concepts and other features.
    It calculates the age and absolute position at death for each patient and
    appends this information to the concepts DataFrame.
    """

    id = "death"
    DEATH_CONCEPT = "Death"

    def create(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        birthdate_col = self.find_column(patients_info, "birth")
        deathdate_col = self.find_column(patients_info, "death")
        logger.info(
            f"Creating death feature using birthdate column: {birthdate_col} and deathdate column: {deathdate_col}"
        )
        patients_info = patients_info[patients_info[deathdate_col].notna()]

        death_info = {"PID": patients_info["PID"].tolist()}
        death_info["CONCEPT"] = [self.DEATH_CONCEPT] * len(patients_info)
        death_info["SEGMENT"] = self._get_last_segments(concepts, patients_info)
        death_info["AGE"] = self._calculate_ages_at_death(
            patients_info, birthdate_col, deathdate_col
        )
        death_info["ABSPOS"] = Utilities.get_abspos_from_origin_point(
            patients_info[deathdate_col], self.config.abspos
        ).to_list()

        # Append death info to concepts
        death_info = pd.DataFrame(death_info)
        return pd.concat([concepts, death_info])

    def _calculate_ages_at_death(
        self, patients_info: pd.DataFrame, birthdate_col: str, deathdate_col: str
    ) -> list:
        ages_at_death = (
            patients_info[deathdate_col] - patients_info[birthdate_col]
        ).dt.days / 365.25
        if self.config.age.get("round"):
            ages_at_death = ages_at_death.round(self.config.age.get("round"))
        return ages_at_death.to_list()

    def _get_last_segments(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> list:
        """For each patient, get the last segment in the concepts DataFrame."""
        if "SEGMENT" not in concepts.columns:
            raise ValueError(
                "Make sure SEGMENT is created before DeathCreator is used."
            )
        last_segments = concepts.groupby("PID")["SEGMENT"].last().to_dict()
        return [last_segments[pid] for pid in patients_info["PID"]]

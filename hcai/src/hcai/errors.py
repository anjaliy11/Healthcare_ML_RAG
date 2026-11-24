"""Typed exceptions. CLI/UI catch HcaiError and render `.args[0]` as a friendly message."""


class HcaiError(Exception):
    """Base class for every expected, user-presentable failure."""


class ConfigError(HcaiError):
    pass


class DataUnavailableError(HcaiError):
    pass


class DataValidationError(HcaiError):
    pass


class LeakageSuspectedError(HcaiError):
    """Test ROC-AUC above the sanity ceiling: stop and investigate."""


class PatientInputError(HcaiError):
    """Patient input failed boundary validation (rejected, never clamped)."""


class ArtifactMissingError(HcaiError):
    pass


class ArtifactIncompatibleError(HcaiError):
    pass


class IngestError(HcaiError):
    pass


class IndexMissingError(HcaiError):
    pass


class IndexCorruptError(HcaiError):
    pass


class InvalidQueryError(HcaiError):
    pass


class LLMError(HcaiError):
    pass

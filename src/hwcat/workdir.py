"""Where every output of a catalogue run goes, under one folder the user picks.

    <root>/era5/<region>_tmax_<year>.nc
    <root>/thresholds/heatwave_threshold_<region>_...nc
    <root>/catalogues/heatwave_events_<region>_<year><tag>.csv
    <root>/catalogues/labels/heatwave_labels_<region>_<year><tag>.npz
    <root>/maps/heatwave_<region>_<year><tag>.pdf
    <root>/robustness/
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workdir:
    root: Path
    region: str = "europe"

    @property
    def era5(self) -> Path:
        return self.root / "era5"

    @property
    def thresholds(self) -> Path:
        return self.root / "thresholds"

    @property
    def catalogues(self) -> Path:
        return self.root / "catalogues"

    @property
    def labels(self) -> Path:
        return self.catalogues / "labels"

    @property
    def maps(self) -> Path:
        return self.root / "maps"

    @property
    def robustness(self) -> Path:
        return self.root / "robustness"

    def era5_file(self, year: int) -> Path:
        return self.era5 / f"{self.region}_tmax_{year}.nc"

    def catalogue_file(self, year: int, tag: str) -> Path:
        return self.catalogues / f"heatwave_events_{self.region}_{year}{tag}.csv"

    def labels_file(self, year: int, tag: str) -> Path:
        return self.labels / f"heatwave_labels_{self.region}_{year}{tag}.npz"

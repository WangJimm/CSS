"""Immutable per-run environment selection; legacy config globals stay unchanged."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from css.experiment import config

ENVIRONMENT_NAMES = ('env', 'new_env')

@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    root: Path
    training_flows: tuple[int, ...]
    validation_flows: tuple[int, ...]
    testing_flows: tuple[int, ...]
    stations: tuple[str, ...]
    manifest_based: bool = False

    @property
    def cfg_path(self): return self.root / 'osm.sumocfg'
    @property
    def net_path(self): return self.root / 'env_static_tls.net.xml'
    @property
    def output_dir(self):
        return self.root.parent / 'outputs' if self.name == 'env' else self.root.parent / 'outputs' / self.name
    @property
    def log_dir(self):
        return self.root.parent / 'logs' if self.name == 'env' else self.root.parent / 'logs' / self.name
    @property
    def model_dir(self): return self.root.parent / 'parameters' / self.name

    def flows(self, split):
        if split not in ('training', 'validation', 'testing'):
            raise ValueError(f'Unknown dataset split: {split}')
        return getattr(self, f'{split}_flows')


def resolve_environment(name='env', project_root=None):
    if name not in ENVIRONMENT_NAMES:
        raise ValueError(f'Unknown environment: {name}; choose from {ENVIRONMENT_NAMES}')
    project = Path(project_root) if project_root is not None else config.HERE
    root = project / name
    if name == 'env':
        return EnvironmentConfig(name, root, tuple(config.training_veh_flows), tuple(config.validation_veh_flows),
                                 tuple(config.testing_veh_flows), ('42496394', '42490303', '42474000'))
    location_file = root / 'cs_locations.json'
    stations = json.loads(location_file.read_text(encoding='utf-8'))
    if not isinstance(stations, list) or not stations or any(not isinstance(s, str) or not s for s in stations) or len(set(stations)) != len(stations):
        raise ValueError(f'Charging station file must contain unique junction IDs: {location_file}')
    return EnvironmentConfig(name, root, (2650, 2800, 2950), (2650, 2800, 2950), (2650, 2800, 2950), tuple(stations), True)

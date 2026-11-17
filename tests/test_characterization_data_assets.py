from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = PROJECT_ROOT / "env"


def ev_count(path: Path) -> int:
    return sum(
        (vehicle.get("id") or "").startswith("EV")
        for vehicle in ET.parse(path).getroot().findall("vehicle")
    )


class DataAssetCharacterizationTests(unittest.TestCase):
    def test_available_seed_ranges_are_stable(self):
        expected = {
            "training": (1, 20, 80),
            "testing": (51, 60, 80),
            "validation": (201, 210, 20),
        }
        for split, (minimum, maximum, count) in expected.items():
            files = list((ENV_ROOT / f"{split}_data").glob("ev_*.rou.xml"))
            seeds = [int(path.stem.rsplit("_", 1)[1].split(".", 1)[0]) for path in files]
            with self.subTest(split=split):
                self.assertEqual((min(seeds), max(seeds), len(files)), (minimum, maximum, count))

    def test_all_power_files_match_route_ev_cardinality(self):
        mismatches = []
        for split in ("training", "testing", "validation"):
            route_dir = ENV_ROOT / f"{split}_data"
            power_dir = ENV_ROOT / f"{split}_data_power"
            for route_path in sorted(route_dir.glob("ev_*.rou.xml")):
                scenario = route_path.name[len("ev_") : -len(".rou.xml")]
                power_path = power_dir / f"power_ev_{scenario}.json"
                powers = json.loads(power_path.read_text(encoding="utf-8"))
                if ev_count(route_path) != len(powers):
                    mismatches.append((split, route_path.name, ev_count(route_path), len(powers)))

        self.assertEqual(mismatches, [])

    def test_default_sumo_configuration_route_exists(self):
        config_path = ENV_ROOT / "osm.sumocfg"
        root = ET.parse(config_path).getroot()
        configured = root.find("./input/route-files").get("value").strip()

        self.assertTrue((ENV_ROOT / configured).is_file())


if __name__ == "__main__":
    unittest.main()

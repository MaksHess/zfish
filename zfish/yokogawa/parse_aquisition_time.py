import datetime
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from pathlib import Path

MEASUREMENT_DATA_FILE = "MeasurementData.mlf"
MEASUREMENT_DETAIL_FILE = "MeasurementDetail.mrf"
ROW_MAP = {k: v for k, v in zip(range(1, 9), "ABCDEFGH")}
PREFIX = "{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}"
XML_NAMESPACES = {"bts": "http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"}

Well = namedtuple("Well", ["r", "c"])
Position = namedtuple("Position", ["x", "y"])


def parse_acquisition_timestamps_with_rewells(flds: list[str]) -> dict[dict[list]]:
    """
    Parse multiple CV8K folders by passing them as a list. Sites are matched based on their well and coordinates.
    If a well was imaged multiple times the last folder in `flds` is used."""
    all_sites = {}
    for fld in flds:
        all_sites = {**all_sites, **parse_acquisition_timestamps(fld)}
    return all_sites


def parse_acquisition_timestamps(fld: str) -> dict[dict[tuple]]:
    fld = Path(fld)
    tree = ET.parse(fld / MEASUREMENT_DATA_FILE)
    root = tree.getroot()
    sites = defaultdict(lambda: defaultdict(tuple))

    for measurement_record in list(root):
        measurement_record_clean = {
            k[len(PREFIX) :]: v for k, v in measurement_record.items()
        }
        well = Well(
            int(measurement_record_clean["Row"]),
            int(measurement_record_clean["Column"]),
        )
        pos = Position(
            float(measurement_record_clean["X"]), float(measurement_record_clean["Y"])
        )
        channel = measurement_record_clean["Ch"]
        time = datetime.datetime.fromisoformat(measurement_record_clean["Time"])

        sites[(well, pos)][channel] = time
    return sites


def rename_sites(sites: dict) -> dict:
    out_sites = {}
    for k in sorted(sites.keys()):
        well_tuple, pos_tuple = k
        v = sites[k]
        out_sites[
            (
                f"{well_to_filename(well_tuple)}",
                f"px{round(pos_tuple[0]):+05d}_py{round(pos_tuple[1]):+05d}",
            )
        ] = v
    return out_sites


def well_to_filename(well_tuple: tuple[int, int]):
    well_row, well_column = well_tuple
    return f"{ROW_MAP[well_row]}{well_column:02d}"


def load_channel_map(fld):
    fn = list(Path(fld).glob("*.mes"))[0]
    tree = ET.parse(fn)
    root = tree.getroot()

    channel_map = {}
    for e in root.find("bts:ChannelList", XML_NAMESPACES):
        channel_map[e.attrib[PREFIX + "Ch"]] = int(re.sub("[^0-9]", "", e[0].text))
    return channel_map

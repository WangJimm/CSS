from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List


def save_routes_xml(
        ev_routes,
        filename,
        start_depart= 13.13,
        depart_gap = 5.0,
        vtype_id = None,
        using_env = " ",
    ):

    filename = f"./{using_env}/new_training_data_3/{filename}_training_set.rou.xml"
    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    for i, edges in enumerate(ev_routes, start=1):
        veh_id = f"EV_{i}"
        depart_time = start_depart + (i - 1) * depart_gap

        veh_attrs = {
            "id": veh_id,
            "depart": f"{depart_time:.2f}",
            "departLane": "best",
            "departSpeed": "max",
        }
        # 如果你有定義 <vType id="ev" ...>，可以在這裡指定 type
        if vtype_id is not None:
            veh_attrs["type"] = vtype_id   # 對應 <vehicle type="ev">

        veh_elem = ET.SubElement(routes, "vehicle", veh_attrs)

        # 支援：edges 可能是 str 或 list
        if isinstance(edges, (list, tuple)):
            edges_str = " ".join(edges)
        else:
            edges_str = str(edges)

        ET.SubElement(veh_elem, "route", {"edges": edges_str})

    tree = ET.ElementTree(routes)

    # 讓輸出的 XML 有縮排（Python 3.9+）
    try:
        ET.indent(tree, space="    ", level=0)
    except AttributeError:
        # 如果是舊版 Python 沒有 indent，就略過，單行也能用
        pass

    # 寫檔，加上 XML 宣告
    tree.write(filename, encoding="UTF-8", xml_declaration=True)

    return


def count_evs_in_rou(rou_path: str, prefix: str = "EV") -> int:
    root = ET.parse(rou_path).getroot()
    n = 0
    for veh in root.findall("vehicle"):
        vid = veh.get("id", "")
        if vid.startswith(prefix):
            n += 1
    return n


def get_evs_in_rou(rou_path: str, prefix: str = "EV") -> list[str]:
    root = ET.parse(rou_path).getroot()
    evs = []
    for veh in root.findall("vehicle"):
        vid = veh.get("id", "")
        if vid.startswith(prefix):
            evs.append(vid)
    return evs


def _split_edges(edges_str: str) -> List[str]:
    return [e for e in edges_str.strip().split() if e]


def change_route(
    rou_in: str,
    rou_out: str,
    target_vid: str,
    min_edges_left: int = 1,
) -> bool:
    """
    只針對 target_vid，把 <route edges="..."> 的第一個 edge 拿掉，另存成 rou_out。
    回傳 True 代表成功修改；False 代表找不到車或不符合條件（例如 edge 太短）。
    """
    tree = ET.parse(rou_in)
    root = tree.getroot()

    for veh in root.findall("vehicle"):
        if veh.get("id") != target_vid:
            continue

        r = veh.find("route")
        if r is None:
            return False

        edges_str = (r.get("edges") or "").strip()
        edges: List[str] = [e for e in edges_str.split() if e]

        # 至少要能刪掉一個且剩下 min_edges_left
        if len(edges) <= min_edges_left:
            return False

        new_edges = edges[1:]
        if len(new_edges) < min_edges_left:
            return False

        r.set("edges", " ".join(new_edges))
        tree.write(rou_out, encoding="utf-8", xml_declaration=True)
        return True

    return False


def keep_first_edge_in_rou_file(rou_in: str, rou_out: str) -> tuple[int, int]:
    tree = ET.parse(rou_in)
    root = tree.getroot()
    modified = 0
    skipped = 0

    for veh in root.findall("vehicle"):
        r = veh.find("route")
        if r is None:
            skipped += 1
            continue
        edges_str = (r.get("edges") or "").strip()
        if not edges_str:
            skipped += 1
            continue
        edges = [e for e in edges_str.split() if e]
        if not edges:
            skipped += 1
            continue
        r.set("edges", edges[0])
        modified += 1

    tree.write(rou_out, encoding="utf-8", xml_declaration=True)
    return modified, skipped


def batch_keep_first_edge( pattern: str = "*.rou.xml", suffix: str = "_firstedge", using_env = None, 
                          general_veh_flows = ["free_0900", "norm_1500", "busy_2100", "heavy_2400"]):
    # files = glob.glob(os.path.join(folder, pattern))
    # print(len(files))
    # print(files[0])
    for rd3 in range(100,151):
        for rd4 in range(1,3):
            for rd5 in range(0,4):
                ev_routes = str(f"./{using_env}/training_data_v2/ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.rou.xml")
                ev_routes_new = str(f"./{using_env}/new_training_data_2/ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.rou.xml")

                keep_first_edge_in_rou_file(ev_routes, ev_routes_new)

    print("done")


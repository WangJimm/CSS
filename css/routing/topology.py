from __future__ import annotations

from collections import deque

MIN_DECISION_LEN = 2.0
MAX_SKIP_HOPS = 8
VCLASS_FILTER = None


def is_external_node(n):
    return (n.getType() != "internal") and (not n.getID().startswith(":"))


def is_external_edge(e):
    f = e.getFunction()
    return (f != "internal") and (not e.getID().startswith(":"))


def is_decisionable_edge(e):
    try:
        return (not e.isInternal()) and (e.getLength() >= MIN_DECISION_LEN)
    except Exception:
        return False


def edge_allows(e, vclass):
    if vclass is None:
        return True
    # sumolib 的 allows 接 lane 為主；這裡簡化為只看第一個 lane
    try:
        ln = e.getLanes()[0]
        return ln.allows(vclass)
    except Exception:
        return True


def expand_to_decision_edges(net, start_edge,
                             max_hops=MAX_SKIP_HOPS,
                             vclass=VCLASS_FILTER):
    """
    從 start_e (Edge 物件，可能是 internal/超短) 開始，往前穿越，
    找到「第一層可決策」的邊集合。
    回傳 list[ (target_node_id, first_real_edge_id, full_prefix_edges) ]
    其中 full_prefix_edges 包含 start_edge_id 以及所有被穿越的邊（不含 first_real_edge 本身）。
    """
    
    start_id = start_edge.getID()
    results = []
    q = deque([(start_edge, [])])  # (當前 Edge 物件, prefix_ids)
    visited  = {start_id}
    hops = 0

    while q and hops <= max_hops:
        hops += 1
        for _ in range(len(q)):
            cur_e, prefix = q.popleft()

            # 若當前邊已可決策，且我們確實穿越過至少一條(=prefix非空)，收斂
            if is_decisionable_edge(cur_e) and len(prefix) > 0:
                tgt = cur_e.getToNode().getID()
                results.append((tgt, cur_e.getID(), prefix.copy()))
                continue

            # 繼續往下一層
            nxt_node = cur_e.getToNode()
            for ne in nxt_node.getOutgoing():
                if vclass is not None and not edge_allows(ne, vclass):
                    continue
                ne_id = ne.getID()
                if ne_id in visited:
                    continue
                visited.add(ne_id)
                q.append((ne, prefix + [cur_e.getID()]))

    # 若完全找不到(例如一路都是 internal 且沒有外側邊)，退而求其次
    if not results:
        tgt = start_edge.getToNode().getID()
        results.append((tgt, start_id, []))

    # 依 first_real_edge 去重
    uniq = {}
    for tgt, first_real, pref in results:
        uniq[first_real] = (tgt, first_real, pref)
    return list(uniq.values())


def build_macro_candidates(net, cur_edge_id, outs_edges):
    """
    outs_edges: List[Edge]  # 你的 candidate_next_junctions 第二個回傳值
    回傳 macros: List[dict]
      - node: 展開後第一個可決策邊的 toNode (作為「下一步節點」)
      - first_edge: 需要 via / setRoute 的第一個可決策邊 ID
      - prefix: 需要先穿越的短邊/內部邊 IDs（可能為空）
    """
    macros = []
    for eout in outs_edges:
        if is_decisionable_edge(eout):
            # eout 本身就可決策，直接當作一個巨集
            macros.append({
                "node": eout.getToNode().getID(),
                "first_edge": eout.getID(),
                "prefix": []
            })
        else:
            # 展開：把 eout 穿越成多個可決策邊
            exps = expand_to_decision_edges(net, eout)
            for tgt_node, first_real, prefix in exps:
                macros.append({
                    "node": tgt_node,
                    "first_edge": first_real,
                    "prefix": prefix
                })
    # 依 first_edge 去重（避免同一條邊重複）
    seen = set()
    uniq_macros = []
    for m in macros:
        key = m["first_edge"]
        if key in seen:
            continue
        seen.add(key)
        uniq_macros.append(m)
    return uniq_macros


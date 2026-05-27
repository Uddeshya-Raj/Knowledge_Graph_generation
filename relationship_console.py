import json
import math
from collections import defaultdict
import pandas as pd
import gradio as gr

# ===================== LOAD =====================
with open("json_files/relation_results.json") as f:
    relations = json.load(f)

with open("json_files/entity_info.json") as f:
    entity_info = json.load(f)

with open("json_files/entity_id_dict.json") as f:
    name_to_id = json.load(f)

id_to_types = {
    eid: data["types"]
    for eid, data in entity_info.items()
}

entity_to_types = {
    data["name"]: data["types"]
    for data in entity_info.values()
}

all_entities = sorted(entity_to_types.keys())

# ===================== TAXONOMY =====================
PARENT = {
    "day": "Time","generic_date": "date","special_date": "date","date": "Time",
    "month": "Time","moon_phase": "Time","measurement_unit": "Time",
    "measurement_method": "Time","nakshatra": "Time",

    "festival": "Activity","ritual_activity": "Activity","mundane_activity": "Activity",

    "abstract_concept": "Concept","attribute": "Concept","state": "Concept",
    "action_process": "Concept","measure_quantity": "Concept","knowledge_linguistic": "Concept",

    "landform": "Geographical_Feature","water_body": "Geographical_Feature",
    "vegetation_region": "Geographical_Feature","atmospheric_region": "Geographical_Feature",

    "human_generic": "Living_Being","human_individual": "Living_Being",
    "animal": "Living_Being","plant": "Living_Being","mythical_living_being": "Living_Being",

    "disease": "Medical_Concept","symptom_physical": "Medical_Concept",
    "symptom_mental": "Medical_Concept","secretion_internal": "Medical_Concept",
    "secretion_external": "Medical_Concept","remedy": "Medical_Concept",

    "metaphysical_entity": "Mythical_Entity","deity": "Mythical_Entity",
    "avatar": "Mythical_Entity","being_class": "Mythical_Entity",
    "individual_figure": "Mythical_Entity","mythical_creature_object": "Mythical_Entity",

    "celestial_phenomenon": "Phenomenon","season": "Phenomenon","natural_phenomenon": "Phenomenon",
}

def is_ancestor(child, parent):
    while child in PARENT:
        if PARENT[child] == parent:
            return True
        child = PARENT[child]
    return False

# ===================== UNIQUE TRIPLES =====================
unique_triples = set()

for item in relations:
    rel_data = item["relation_data"]
    if rel_data["relation"] == "NONE":
        continue
    unique_triples.add((
        rel_data["source_entity"],
        rel_data["relation"],
        rel_data["target_entity"]
    ))

all_relations = sorted(list(set(r for (_, r, _) in unique_triples)))

# ===================== TYPE GRAPH =====================
type_rel_counts = defaultdict(int)
type_rel_instances = defaultdict(set)

MAX_TYPES = 3

for (src, rel, tgt) in unique_triples:

    if src not in name_to_id or tgt not in name_to_id:
        continue

    src_types = id_to_types.get(name_to_id[src], [])[:MAX_TYPES]
    tgt_types = id_to_types.get(name_to_id[tgt], [])[:MAX_TYPES]

    for t1 in src_types:
        for t2 in tgt_types:

            if t1 == t2:
                continue

            if is_ancestor(t1, t2) or is_ancestor(t2, t1):
                continue

            key = (t1, rel, t2)

            type_rel_counts[key] += 1
            type_rel_instances[key].add((src, tgt))

all_types = sorted(list(set(
    [t for (t, _, _) in type_rel_counts] +
    [t for (_, _, t) in type_rel_counts]
)))

# ===================== PMI =====================
total = sum(type_rel_counts.values())

type_freq = defaultdict(int)
rel_freq = defaultdict(int)

for (t1, r, t2), c in type_rel_counts.items():
    type_freq[t1] += c
    type_freq[t2] += c
    rel_freq[r] += c

type_rel_pmi = {}

for (t1, r, t2), c in type_rel_counts.items():
    p_xyz = c / total
    p_t1 = type_freq[t1] / total
    p_t2 = type_freq[t2] / total
    p_r = rel_freq[r] / total

    type_rel_pmi[(t1, r, t2)] = math.log(p_xyz / (p_t1 * p_r * p_t2 + 1e-12) + 1e-12)

# ===================== ORIGINAL FUNCTIONS =====================
def aggregate_types(bidirectional=False):
    agg = defaultdict(lambda: {"count": 0, "pmi_sum": 0, "rels": defaultdict(int)})

    for (t1, r, t2), c in type_rel_counts.items():
        key = tuple(sorted([t1, t2])) if bidirectional else (t1, t2)

        agg[key]["count"] += c
        agg[key]["pmi_sum"] += type_rel_pmi[(t1, r, t2)]
        agg[key]["rels"][r] += c

    rows = []
    for (t1, t2), data in agg.items():
        top_rels = sorted(data["rels"].items(), key=lambda x: -x[1])[:3]

        rows.append({
            "type_1": t1,
            "type_2": t2,
            "total_count": data["count"],
            "avg_pmi": round(data["pmi_sum"] / max(1, len(data["rels"])), 3),
            "top_relations": str(top_rels)
        })

    return pd.DataFrame(rows).sort_values(by="avg_pmi", ascending=False)

def relation_view(pmi_thresh, min_count):
    rows = []
    for (t1, r, t2), score in type_rel_pmi.items():
        count = type_rel_counts[(t1, r, t2)]
        if score >= pmi_thresh and count >= min_count:
            rows.append({
                "type_1": t1,
                "relation": r,
                "type_2": t2,
                "PMI": round(score, 3),
                "count": count
            })
    return pd.DataFrame(rows).sort_values(by="PMI", ascending=False)

def instance_view(t1, r, t2):
    pairs = list(type_rel_instances.get((t1, r, t2), set()))
    return pd.DataFrame(pairs, columns=["source", "target"])

# ===================== SEARCH =====================
def entity_search(src_e, src_t, rel, tgt_e, tgt_t):

    src_e = None if src_e in ["", None] else src_e
    src_t = None if src_t in ["", None] else src_t
    rel   = None if rel   in ["", None] else rel
    tgt_e = None if tgt_e in ["", None] else tgt_e
    tgt_t = None if tgt_t in ["", None] else tgt_t

    if not any([src_e, src_t, rel, tgt_e, tgt_t]):
        return pd.DataFrame(columns=["source", "relation", "target"])

    results = []

    for (s, r, t) in unique_triples:

        if src_e is not None and s != src_e:
            continue
        if tgt_e is not None and t != tgt_e:
            continue
        if rel is not None and r != rel:
            continue
        if src_t is not None and src_t not in entity_to_types.get(s, []):
            continue
        if tgt_t is not None and tgt_t not in entity_to_types.get(t, []):
            continue

        results.append({
            "source": s,
            "relation": r,
            "target": t,
            "source_types": entity_to_types.get(s, []),
            "target_types": entity_to_types.get(t, [])
        })

    return pd.DataFrame(results)

# ===================== UI =====================
with gr.Blocks() as demo:

    with gr.Tabs():

        # -------- Explorer --------
        with gr.Tab("Explorer"):
            gr.Markdown("## Ontology Explorer")

            with gr.Row():
                pmi_slider = gr.Slider(0, 5, value=1.0)
                count_slider = gr.Slider(1, 20, value=5)
                bidir = gr.Checkbox(label="Merge directions")

            rel_table = gr.Dataframe(interactive=True)
            agg_table = gr.Dataframe(interactive=True)
            instance_table = gr.Dataframe()

            state = gr.State()

            def refresh(p, c, b):
                r = relation_view(p, c)
                a = aggregate_types(b)
                return r, r, a

            def drill(evt: gr.SelectData, df):
                row = df.iloc[evt.index[0]]
                return instance_view(row["type_1"], row["relation"], row["type_2"])

            pmi_slider.change(refresh, [pmi_slider, count_slider, bidir],
                              [rel_table, state, agg_table])
            count_slider.change(refresh, [pmi_slider, count_slider, bidir],
                                [rel_table, state, agg_table])
            bidir.change(refresh, [pmi_slider, count_slider, bidir],
                         [rel_table, state, agg_table])

            rel_table.select(drill, [state], instance_table)

            demo.load(refresh, [pmi_slider, count_slider, bidir],
                      [rel_table, state, agg_table])

        # -------- Search --------
        with gr.Tab("Search"):

            gr.Markdown("### Leave fields empty = wildcard search")

            src_entity = gr.Dropdown([None] + all_entities, value=None, label="Source Entity")
            src_type = gr.Dropdown([None] + all_types, value=None, label="Source Type")
            rel_dd = gr.Dropdown([None] + all_relations, value=None, label="Relation")
            tgt_entity = gr.Dropdown([None] + all_entities, value=None, label="Target Entity")
            tgt_type = gr.Dropdown([None] + all_types, value=None, label="Target Type")

            btn = gr.Button("Search")
            out = gr.Dataframe()

            btn.click(entity_search,
                      inputs=[src_entity, src_type, rel_dd, tgt_entity, tgt_type],
                      outputs=out)

demo.launch()
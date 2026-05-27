# %%
import json
import os
import sys
import traceback
import numpy as np

from collections import defaultdict
from itertools import combinations

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================

ENTITY_INFO_FILE = "json_files/entity_info.json"
RELATION_FILE = "json_files/relation_results.json"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

TOP_K = 5

DIRECT_THRESHOLD = 0.95
LLM_THRESHOLD = 0.65

# ============================================================
# LLM
# ============================================================

CLIENT = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="none"
)

# ============================================================
# TAXONOMY
# ============================================================

TAXONOMY = {
    "Time": {
        "day": {},
        "date": {
            "special_date": {},
            "generic_date": {}
        },
        "month": {},
        "moon_phase": {},
        "measurement_unit": {},
        "measurement_method": {},
        "nakshatra": {},
        "other": {}
    },

    "Activity": {
        "festival": {},
        "ritual_activity": {},
        "mundane_activity": {},
        "other": {}
    },

    "Medical_Concept": {
        "disease": {},
        "symptom_physical": {},
        "symptom_mental": {},
        "secretion_internal": {},
        "secretion_external": {},
        "remedy": {},
        "other": {}
    },

    "Phenomenon": {
        "celestial_phenomenon": {},
        "season": {},
        "natural_phenomenon": {},
        "other": {}
    },

    "Concept": {
        "abstract_concept": {},
        "attribute": {},
        "state": {},
        "action_process": {},
        "measure_quantity": {},
        "knowledge_linguistic": {},
        "other": {}
    },

    "Geographical_Feature": {
        "landform": {},
        "water_body": {},
        "vegetation_region": {},
        "atmospheric_region": {},
        "other": {}
    },

    "Living_Being": {
        "human_generic": {},
        "human_individual": {},
        "animal": {},
        "plant": {},
        "mythical_living_being": {},
        "other": {}
    },

    "Mythical_Entity": {
        "metaphysical_entity": {},
        "deity": {},
        "avatar": {},
        "being_class": {},
        "individual_figure": {},
        "mythical_creature_object": {},
        "other": {}
    }
}

# ============================================================
# DEBUG
# ============================================================

CURRENT_ENTITY = None
CURRENT_PAIR = None
CURRENT_SIMILARITY = None

# ============================================================
# ERROR HANDLER
# ============================================================

def fatal_error(e):

    print("\n" + "=" * 80)
    print("FATAL ERROR")
    print("=" * 80)

    print("\nERROR:")
    print(type(e).__name__)
    print(str(e))

    print("\nTRACEBACK:")
    traceback.print_exc()

    print("\nCURRENT_ENTITY:")
    print(CURRENT_ENTITY)

    print("\nCURRENT_PAIR:")
    print(CURRENT_PAIR)

    print("\nCURRENT_SIMILARITY:")
    print(CURRENT_SIMILARITY)

    print("\n" + "=" * 80)

    sys.exit(1)

# ============================================================
# BUILD LEAF NODE SET
# ============================================================

LEAF_NODES = set()

def collect_leaf_nodes(subtree):

    for node, child in subtree.items():

        if child:
            collect_leaf_nodes(child)

        else:
            LEAF_NODES.add(node)

collect_leaf_nodes(TAXONOMY)

# ============================================================
# LOAD DATA
# ============================================================

try:

    with open(ENTITY_INFO_FILE, "r", encoding="utf-8") as f:
        ENTITY_INFO = json.load(f)

    if os.path.exists(RELATION_FILE):

        with open(RELATION_FILE, "r", encoding="utf-8") as f:
            RELATIONS = json.load(f)

    else:
        RELATIONS = []

except Exception as e:
    fatal_error(e)

# ============================================================
# EMBEDDING MODEL
# ============================================================

try:

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")

except Exception as e:
    fatal_error(e)

# ============================================================
# SAVE
# ============================================================

def atomic_save(data, path):

    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp, path)

# ============================================================
# GET VALID LEAF TYPES
# ============================================================

def get_leaf_types(entity):

    types = entity.get("types", [])

    valid = []

    for t in types:

        if not isinstance(t, str):
            continue

        t = t.strip()

        if t in LEAF_NODES:

            # ignore "other"
            if t != "other":
                valid.append(t)

    return list(set(valid))

# ============================================================
# EMBEDDINGS
# ============================================================

def build_embedding_text(entity):

    name = entity.get("name", "")
    desc = entity.get("description", "")

    return f"{name} is defined as {desc}"

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

try:

    print("\nGenerating embeddings...")

    ENTITY_EMBEDDINGS = {}

    for node_id, entity in ENTITY_INFO.items():

        leafs = get_leaf_types(entity)

        # ----------------------------------------------------
        # ignore entities with NO leaf types
        # ----------------------------------------------------

        if not leafs:
            continue

        text = build_embedding_text(entity)

        emb = model.encode(
            text,
            normalize_embeddings=True
        )

        ENTITY_EMBEDDINGS[node_id] = emb

    print("Embeddings generated.")

except Exception as e:
    fatal_error(e)

# ============================================================
# PURE TYPE BUCKETS
# ============================================================

try:

    PURE_TYPE_BUCKETS = defaultdict(list)

    for node_id, entity in ENTITY_INFO.items():

        if node_id not in ENTITY_EMBEDDINGS:
            continue

        leafs = get_leaf_types(entity)

        if len(leafs) == 1:

            PURE_TYPE_BUCKETS[leafs[0]].append(node_id)

except Exception as e:
    fatal_error(e)

# ============================================================
# PRIMARY TYPE
# ============================================================

def compute_primary_type(node_id):

    entity = ENTITY_INFO[node_id]

    leafs = get_leaf_types(entity)

    # --------------------------------------------------------
    # ignore entities without leaf types
    # --------------------------------------------------------

    if not leafs:
        return None

    # --------------------------------------------------------
    # single type
    # --------------------------------------------------------

    if len(leafs) == 1:
        return leafs[0]

    emb = ENTITY_EMBEDDINGS[node_id]

    scores = {}

    # --------------------------------------------------------
    # compare against pure type neighborhoods
    # --------------------------------------------------------

    for t in leafs:

        candidates = PURE_TYPE_BUCKETS[t]

        sims = []

        for other_id in candidates:

            if other_id == node_id:
                continue

            sim = cosine_similarity(
                [emb],
                [ENTITY_EMBEDDINGS[other_id]]
            )[0][0]

            sims.append(sim)

        if not sims:
            continue

        sims = sorted(sims, reverse=True)[:TOP_K]

        scores[t] = np.mean(sims)

    # --------------------------------------------------------
    # fallback
    # --------------------------------------------------------

    if not scores:
        return leafs[0]

    return max(scores, key=scores.get)

# ============================================================
# PRIMARY TYPES
# ============================================================

try:

    print("\nComputing primary types...")

    PRIMARY_TYPES = {}

    for node_id in ENTITY_EMBEDDINGS:

        ptype = compute_primary_type(node_id)

        if ptype is not None:
            PRIMARY_TYPES[node_id] = ptype

    print("Primary types computed.")

except Exception as e:
    fatal_error(e)

# ============================================================
# GROUPS
# ============================================================

try:

    GROUPS = defaultdict(list)

    for node_id, ptype in PRIMARY_TYPES.items():

        GROUPS[ptype].append(node_id)

except Exception as e:
    fatal_error(e)

# ============================================================
# LLM
# ============================================================

class SameAsVerification(BaseModel):

    is_same: bool

LLM_PROMPT = """
You are a strict semantic identity verifier.

Determine whether the two entities refer to the EXACT SAME semantic entity.

Rules:
1. Aliases and honorifics may still refer to same entity.
2. Broad semantic similarity is NOT enough.
3. Return true ONLY if both refer to same identity.
4. Do NOT use outside knowledge.

Output JSON only:
{
  "is_same": true/false
}
"""

def call_llm(user_query):

    response = CLIENT.chat.completions.create(
        model="Qwen/Qwen2.5-14B-Instruct",
        messages=[
            {
                "role": "system",
                "content": LLM_PROMPT
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=0.1,
        max_tokens=256,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SameAsVerification",
                "schema": SameAsVerification.model_json_schema()
            }
        }
    )

    return response.choices[0].message.content

# ============================================================
# EXISTING SAME_AS
# ============================================================

try:

    EXISTING_SAME_AS = set()

    for rel in RELATIONS:

        r = rel["relation_data"]

        if r["relation"] == "IS_SAME_AS":

            s = r["source_entity"]
            t = r["target_entity"]

            EXISTING_SAME_AS.add(
                tuple(sorted([s, t]))
            )

except Exception as e:
    fatal_error(e)

# ============================================================
# MAIN LOOP
# ============================================================

try:

    print("\nStarting grouping...")

    for ptype, node_ids in GROUPS.items():

        print("\n" + "=" * 60)
        print(f"GROUP TYPE: {ptype}")
        print(f"ENTITY COUNT: {len(node_ids)}")
        print("=" * 60)

        for n1, n2 in combinations(node_ids, 2):

            e1 = ENTITY_INFO[n1]
            e2 = ENTITY_INFO[n2]

            CURRENT_PAIR = (
                e1["name"],
                e2["name"]
            )

            pair_key = tuple(
                sorted([
                    e1["name"],
                    e2["name"]
                ])
            )

            if pair_key in EXISTING_SAME_AS:
                continue

            emb1 = ENTITY_EMBEDDINGS[n1]
            emb2 = ENTITY_EMBEDDINGS[n2]

            sim = cosine_similarity(
                [emb1],
                [emb2]
            )[0][0]

            CURRENT_SIMILARITY = sim

            print(
                f"{e1['name']} <-> {e2['name']} = {sim:.4f}"
            )

            # =================================================
            # DIRECT SAME_AS
            # =================================================

            if sim >= DIRECT_THRESHOLD:

                relation = {
                    "metadata": {
                        "method": "embedding",
                        "similarity": float(sim),
                        "grouping_class": ptype
                    },

                    "relation_data": {
                        "source_entity": e1["name"],
                        "relation": "IS_SAME_AS",
                        "target_entity": e2["name"]
                    }
                }

                RELATIONS.append(relation)

                EXISTING_SAME_AS.add(pair_key)

                atomic_save(
                    RELATIONS,
                    RELATION_FILE
                )

                print("DIRECT SAME_AS")

            # =================================================
            # LLM CHECK
            # =================================================

            elif sim >= LLM_THRESHOLD:

                user_query = f"""
Entity 1:
{e1['name']}

Description:
{e1.get('description', '')}

Entity 2:
{e2['name']}

Description:
{e2.get('description', '')}
"""

                raw = call_llm(user_query)

                parsed = json.loads(raw)

                if parsed["is_same"]:

                    relation = {
                        "metadata": {
                            "method": "llm_verification",
                            "similarity": float(sim),
                            "grouping_class": ptype
                        },

                        "relation_data": {
                            "source_entity": e1["name"],
                            "relation": "IS_SAME_AS",
                            "target_entity": e2["name"]
                        }
                    }

                    RELATIONS.append(relation)

                    EXISTING_SAME_AS.add(pair_key)

                    atomic_save(
                        RELATIONS,
                        RELATION_FILE
                    )

                    print("LLM VERIFIED SAME_AS")

    print("\nDONE.")

except Exception as e:
    fatal_error(e)
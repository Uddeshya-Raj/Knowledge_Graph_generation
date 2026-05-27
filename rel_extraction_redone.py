# %%
import json
import os
import time
from collections import defaultdict
from openai import OpenAI
from pydantic import BaseModel, Field

# ============================================================
# CONFIG
# ============================================================
CLIENT = OpenAI(
    base_url='http://127.0.0.1:8001/v1',
    api_key='none'
)

STATE_FILE = "json_files/relation_state.json"
RESULTS_FILE = "json_files/relation_results.json"

ENTITY_INFO_FILE = "json_files/entity_info.json"
ENTITY_ID_FILE = "json_files/entity_id_dict.json"

RELATION_INDEX_FILE = "json_files/relation_type_index.json"

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
# BUILD LEAF/PARENT STRUCTURES
# ============================================================

LEAF_NODES = set()
PARENT_TO_CHILDREN = defaultdict(set)
CHILD_TO_PARENT = {}

def traverse(node, subtree):
    for child, sub in subtree.items():

        PARENT_TO_CHILDREN[node].add(child)
        CHILD_TO_PARENT[child] = node

        if sub:
            traverse(child, sub)
        else:
            LEAF_NODES.add(child)

for root, subtree in TAXONOMY.items():
    traverse(root, subtree)

# ============================================================
# ATOMIC SAVE
# ============================================================

def atomic_save(data, filename):
    tmp = filename + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp, filename)

# ============================================================
# LOAD / SAVE STATE
# ============================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "chapter": 0,
            "group": 0,
            "sentence": 0,
            "i": 0,
            "j": 1,
            "direction": 0
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(chapter, group, sentence, i, j, direction):

    state = {
        "chapter": chapter,
        "group": group,
        "sentence": sentence,
        "i": i,
        "j": j,
        "direction": direction
    }

    atomic_save(state, STATE_FILE)

# ============================================================
# RESULTS
# ============================================================

def load_results():

    if not os.path.exists(RESULTS_FILE):
        return []

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(results):
    atomic_save(results, RESULTS_FILE)

# ============================================================
# LOAD ENTITY DATA
# ============================================================

with open(ENTITY_INFO_FILE, "r", encoding="utf-8") as f:
    ENTITY_INFO = json.load(f)

with open(ENTITY_ID_FILE, "r", encoding="utf-8") as f:
    ENTITY_ID_MAP = json.load(f)

# ============================================================
# RELATION INDEX
# ============================================================

if os.path.exists(RELATION_INDEX_FILE):

    with open(RELATION_INDEX_FILE, "r", encoding="utf-8") as f:
        RELATION_INDEX = json.load(f)

else:
    RELATION_INDEX = {}

# ============================================================
# TYPE FILTERING LOGIC
# ============================================================

def get_matching_types(entity_name):

    node_id = ENTITY_ID_MAP.get(entity_name)

    if not node_id:
        return []

    all_types = ENTITY_INFO[node_id].get("types", [])

    final_types = set()

    for t in all_types:

        # ----------------------------------------
        # LEAF NODE
        # ----------------------------------------
        if t in LEAF_NODES:

            # if leaf is "other"
            if t == "other":

                parent = CHILD_TO_PARENT.get(t)

                if parent:
                    final_types.add(parent)

            else:
                final_types.add(t)

    return sorted(final_types)

# ============================================================
# PRIOR RELATIONS
# ============================================================

def get_prior_relations(types1, types2):

    relations = set()

    for t1 in types1:
        for t2 in types2:

            k1 = f"{t1}|{t2}"
            k2 = f"{t2}|{t1}"

            if k1 in RELATION_INDEX:
                relations.update(RELATION_INDEX[k1])

            if k2 in RELATION_INDEX:
                relations.update(RELATION_INDEX[k2])

    return sorted(relations)

# ============================================================
# LLM CALL
# ============================================================

def call_llm(
    system_prompt,
    user_query,
    model_name="Qwen/Qwen2.5-14B-Instruct",
    response_model=None
):

    api_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    if response_model:

        api_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
            },
        }

    response = CLIENT.chat.completions.create(**api_params)

    return response.choices[0].message.content

# ============================================================
# SCHEMA
# ============================================================

class EntityRelation(BaseModel):

    scratchpad: str = Field(
        description="Reasoning to verify whether Entity 1 relation Entity 2 is directly supported by the sentence."
    )

    source_entity: str = Field(
        description="Must EXACTLY match Entity 1"
    )

    relation: str = Field(
        description="snake_case relation or NONE"
    )
    
    target_entity: str = Field(
        description="Must EXACTLY match Entity 2"
    )


# ============================================================
# LOAD PROMPT
# ============================================================

with open(
    "prompts/relation_extraction_redone.txt",
    "r",
    encoding="utf-8"
) as f:

    SYSTEM_PROMPT = f.read()

# ============================================================
# INIT
# ============================================================

state = load_state()
results_data = load_results()

start_chapter = state["chapter"]
start_group = state["group"]
start_sentence = state["sentence"]
start_i = state["i"]
start_j = state["j"]
start_direction = state["direction"]

print(
    "Resuming from:",
    f"chapter={start_chapter},",
    f"group={start_group},",
    f"sentence={start_sentence},",
    f"i={start_i},",
    f"j={start_j},",
    f"dir={start_direction}"
)

# ============================================================
# MAIN LOOP
# ============================================================

NUM_CHAPTERS = 10

for c in range(NUM_CHAPTERS):

    if c < start_chapter:
        continue

    file_path = f"./entity_output_files_redone/chapter_{c}_output.json"

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        chapter = json.load(f)

    for x, group in enumerate(chapter):

        if c == start_chapter and x < start_group:
            continue

        for y, data in enumerate(group["extracted_entities"][:10]):

            if (
                c == start_chapter
                and x == start_group
                and y < start_sentence
            ):
                continue

            entities = data.get("entities", [])

            if not entities:
                continue

            sentence_text = data["sentence"]

            for i in range(len(entities)):

                if (
                    c == start_chapter
                    and x == start_group
                    and y == start_sentence
                    and i < start_i
                ):
                    continue

                for j in range(i + 1, len(entities)):

                    if (
                        c == start_chapter
                        and x == start_group
                        and y == start_sentence
                        and i == start_i
                        and j < start_j
                    ):
                        continue

                    for direction in [0, 1]:

                        if (
                            c == start_chapter
                            and x == start_group
                            and y == start_sentence
                            and i == start_i
                            and j == start_j
                            and direction < start_direction
                        ):
                            continue

                        # -----------------------------------
                        # DIRECTION
                        # -----------------------------------

                        if direction == 0:
                            e1, e2 = entities[i], entities[j]
                        else:
                            e1, e2 = entities[j], entities[i]

                        # -----------------------------------
                        # TYPES
                        # -----------------------------------

                        e1_types = get_matching_types(e1)
                        e2_types = get_matching_types(e2)

                        # -----------------------------------
                        # PRIOR RELATIONS
                        # -----------------------------------

                        prior_relations = get_prior_relations(
                            e1_types,
                            e2_types
                        )

                        # -----------------------------------
                        # USER QUERY
                        # -----------------------------------

                        user_query = (
                            f"Sentence:\n"
                            f"{sentence_text}\n\n"

                            f"Entity 1:\n"
                            f"{e1}\n\n"

                            f"Entity 1 Types:\n"
                            f"{json.dumps(e1_types, ensure_ascii=False)}\n\n"

                            f"Entity 2:\n"
                            f"{e2}\n\n"

                            f"Entity 2 Types:\n"
                            f"{json.dumps(e2_types, ensure_ascii=False)}\n\n"

                            f"Previously observed relations:\n"
                            f"{json.dumps(prior_relations, ensure_ascii=False)}"
                        )

                        try:

                            response = call_llm(
                                SYSTEM_PROMPT,
                                user_query,
                                response_model=EntityRelation
                            )

                            print(
                                f"[c={c}, g={x}, s={y}] "
                                f"({i},{j}) dir={direction}"
                            )

                            print(response)
                            print("-" * 60)

                            relation_dict = json.loads(response)

                            relation_dict.pop("scratchpad", None)

                            # -----------------------------------
                            # SAVE RESULT
                            # -----------------------------------

                            record = {
                                "metadata": {
                                    "chapter": c,
                                    "group": x,
                                    "sentence_idx": y,
                                    "i": i,
                                    "j": j,
                                    "direction": direction
                                },

                                "relation_data": relation_dict
                            }

                            results_data.append(record)

                            # -----------------------------------
                            # UPDATE RELATION INDEX
                            # -----------------------------------

                            relation = relation_dict["relation"]

                            if relation != "NONE":

                                for t1 in e1_types:
                                    for t2 in e2_types:

                                        key = f"{t1}|{t2}"

                                        if key not in RELATION_INDEX:
                                            RELATION_INDEX[key] = []

                                        if relation not in RELATION_INDEX[key]:
                                            RELATION_INDEX[key].append(relation)

                                atomic_save(
                                    RELATION_INDEX,
                                    RELATION_INDEX_FILE
                                )

                        except Exception as e:

                            print("Error:", e)
                            time.sleep(2)

                        # -----------------------------------
                        # CHECKPOINT
                        # -----------------------------------

                        save_state(
                            c,
                            x,
                            y,
                            i,
                            j,
                            direction
                        )

                        save_results(results_data)

print("Processing complete.")
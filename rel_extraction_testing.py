# %%
import json
import os
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

# %%
CLIENT = OpenAI(base_url='http://127.0.0.1:8001/v1', api_key='none')

STATE_FILE = "json_files/relation_state_no_schema.json"
RESULTS_FILE = "json_files/relation_results_no_schema.json"


# ------------------------------------------------------------------
# Atomic save
# ------------------------------------------------------------------
def atomic_save(data, filename):
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, filename)


# ------------------------------------------------------------------
# Load / Save state and results
# ------------------------------------------------------------------
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


def load_results():
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(results):
    atomic_save(results, RESULTS_FILE)


# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------
def call_llm(system_prompt, user_query, model_name="Qwen/Qwen2.5-14B-Instruct", response_model=None):
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


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------
class EntityRelation(BaseModel):
    scratchpad: str = Field(description="Thought process to find and cross-check if Entity_1 -> relation -> Entity_2 is a valid triplet supported by text or in case of NONE there really isn't any connection between entities.")
    source_entity: str = Field(description="Must EXACTLY match Entity 1")
    target_entity: str = Field(description="Must EXACTLY match Entity 2")
    relation: str = Field(description="snake_case or NONE")


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------
state = load_state()
results_data = load_results()

start_chapter = state["chapter"]
start_group = state["group"]
start_sentence = state["sentence"]
start_i = state["i"]
start_j = state["j"]
start_direction = state["direction"]

print("Resuming from:",
      f"chapter={start_chapter}, group={start_group}, sentence={start_sentence}, i={start_i}, j={start_j}, dir={start_direction}")


# ------------------------------------------------------------------
# Load prompt once
# ------------------------------------------------------------------
with open("prompts/relation_extraction.txt", 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()


# ------------------------------------------------------------------
# Main loop (CHAPTER-WISE)
# ------------------------------------------------------------------
NUM_CHAPTERS = 10  # 0 to 24

for c in range(0, NUM_CHAPTERS):

    if c < start_chapter:
        continue

    file_path = f"./entity_output_files_redone/chapter_{c}_output.json"

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

    with open(file_path, "r", encoding='utf-8') as f:
        chapter = json.load(f)

    for x, group in enumerate(chapter):

        if c == start_chapter and x < start_group:
            continue

        for y, data in enumerate(group['extracted_entities'][:10]):

            if c == start_chapter and x == start_group and y < start_sentence:
                continue

            entities = data.get('entities', [])
            if not entities:
                continue

            sentence = f"Sentence:\n{data['sentence']}\n"

            for i in range(len(entities)):

                if (
                    c == start_chapter and
                    x == start_group and
                    y == start_sentence and
                    i < start_i
                ):
                    continue

                for j in range(i + 1, len(entities)):

                    if (
                        c == start_chapter and
                        x == start_group and
                        y == start_sentence and
                        i == start_i and
                        j < start_j
                    ):
                        continue

                    for direction in [0, 1]:

                        if (
                            c == start_chapter and
                            x == start_group and
                            y == start_sentence and
                            i == start_i and
                            j == start_j and
                            direction < start_direction
                        ):
                            continue

                        if direction == 0:
                            e1, e2 = entities[i], entities[j]
                        else:
                            e1, e2 = entities[j], entities[i]

                        user_query = (
                            f"{sentence}"
                            f"Entity 1: {e1}\n"
                            f"Entity 2: {e2}\n"
                        )

                        try:
                            response = call_llm(
                                SYSTEM_PROMPT,
                                user_query,
                                response_model=EntityRelation
                            )
                            
                            # Log to console
                            print(f"[c={c}, g={x}, s={y}] ({i},{j}) dir={direction}")
                            print(response)
                            print("----------------------------------")

                            # Parse the JSON response from the LLM
                            relation_dict = json.loads(response)
                            relation_dict.pop("scratchpad", None)

                            # Append the structured record to our results list
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

                        except Exception as e:
                            print("Error:", e)
                            time.sleep(2)

                        # 🔥 CRITICAL CHECKPOINT
                        # Save the updated progress and the continuously growing results file
                        save_state(c, x, y, i, j, direction)
                        save_results(results_data)

print("Processing complete.")
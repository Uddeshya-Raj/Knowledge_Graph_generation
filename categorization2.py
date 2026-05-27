import json
import time
import os
from collections import defaultdict
from typing import List, Optional, Type

from openai import OpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

CLIENT = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="none")

STATE_FILE = "processing_state.json"


# ---------------------------------------------------------------------------
# Atomic save (CRITICAL)
# ---------------------------------------------------------------------------

def atomic_save(data, filename):
    temp_file = filename + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(temp_file, filename)


# ---------------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------------

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "chapter": 0,
            "group": 0,
            "sentence": -1,  # start BEFORE first sentence
            "registry": defaultdict(set),
            "failed_sentences": [],
            "validation_flags": []
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    return {
        "chapter": state["last_processed"]["chapter"],
        "group": state["last_processed"]["group"],
        "sentence": state["last_processed"]["sentence"],
        "registry": defaultdict(set, {
            k: set(v) for k, v in state["global_entity_registry"].items()
        }),
        "failed_sentences": state["failed_sentences"],
        "validation_flags": state["validation_flags"]
    }


def save_state(registry, failed, flags, chapter, group, sentence):
    state = {
        "last_processed": {
            "chapter": chapter,
            "group": group,
            "sentence": sentence
        },
        "global_entity_registry": {
            k: list(v) for k, v in registry.items()
        },
        "failed_sentences": failed,
        "validation_flags": flags
    }

    atomic_save(state, STATE_FILE)


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class ExtractedEntities(BaseModel):
    # 🔥 CRITICAL: The scratchpad MUST be the first field.
    step_by_step_reasoning: str = Field(
        description="Think step-by-step. Read the sentence, check the formal definitions, and explain which entities belong to which category before listing them."
    )
    
    Mythical_Entity: Optional[List[str]] = Field(default=None)
    Celestial_Entity: Optional[List[str]] = Field(default=None)
    Phenomenon: Optional[List[str]] = Field(default=None)
    Time: Optional[List[str]] = Field(default=None)
    Food: Optional[List[str]] = Field(default=None)
    Activity: Optional[List[str]] = Field(default=None)
    Concept: Optional[List[str]] = Field(default=None)
    Object: Optional[List[str]] = Field(default=None)
    Living_Being: Optional[List[str]] = Field(default=None)
    Text: Optional[List[str]] = Field(default=None)
    Location: Optional[List[str]] = Field(default=None)
    Primordial_Element: Optional[List[str]] = Field(default=None)
    Medical_Concept: Optional[List[str]] = Field(default=None)
    Geographical_Feature: Optional[List[str]] = Field(default=None)
    Event: Optional[List[str]] = Field(default=None)
    Emotions: Optional[List[str]] = Field(default=None)
    Body_part: Optional[List[str]] = Field(default=None)
    Sanskrit_text: Optional[List[str]] = Field(default=None)
    Other: Optional[List[str]] = Field(default=None)

    Social_Group_and_Role: Optional[List[str]] = Field(
        default=None,
        alias="Social_Group_&_Role"
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_llm(system_prompt, user_query, response_model=None):
    api_params = {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": 8192,
        "temperature": 0.2,
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


# ---------------------------------------------------------------------------
# Load system prompt
# ---------------------------------------------------------------------------

with open("prompts/recategorization_lvl_1.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# ---------------------------------------------------------------------------
# Load previous state
# ---------------------------------------------------------------------------

state = load_state()

start_chapter = state["chapter"]
start_group = state["group"]
start_sentence = state["sentence"]

global_entity_registry = state["registry"]
failed_sentences = state["failed_sentences"]
validation_flags = state["validation_flags"]

global_expected_entities = set()
max_retries = 3

print("Resuming from:",
      f"Chapter {start_chapter}, Group {start_group}, Sentence {start_sentence}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for i in range(5):

    if i < start_chapter:
        continue

    try:
        with open(f"entity_output_files/chapter_{i}_output.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        failed_sentences.append({"chapter": i, "reason": "file_not_found"})
        continue

    for group_idx, group in enumerate(data):

        if i == start_chapter and group_idx < start_group:
            continue

        for sent_idx, sentence_obj in enumerate(group.get("extracted_entities", [])):

            if (
                i == start_chapter
                and group_idx == start_group
                and sent_idx <= start_sentence
            ):
                continue

            expected_entities = set(sentence_obj.get("entities", []))
            global_expected_entities.update(expected_entities)

            if not expected_entities:
                continue

            base_user_query = json.dumps(sentence_obj, ensure_ascii=False, indent=4)
            prompt_suffix = ""
            success = False

            for attempt in range(max_retries):
                try:
                    raw_response = call_llm(
                        SYSTEM_PROMPT,
                        base_user_query + prompt_suffix,
                        ExtractedEntities,
                    )

                    parsed = ExtractedEntities.model_validate_json(raw_response)
                    
                    # 🔥 THE CHANGE IS HERE: Add the exclude parameter
                    category_dict = parsed.model_dump(by_alias=True, exclude={"step_by_step_reasoning"})

                    returned_entities = set()
                    for v in category_dict.values():
                        if v:
                            returned_entities.update(v)

                    missing = expected_entities - returned_entities

                    if missing and attempt < max_retries - 1:
                        prompt_suffix = f"\n\nMissing: {list(missing)}"
                        raise ValueError("Retrying due to missing entities")

                    for k, v in category_dict.items():
                        if v:
                            global_entity_registry[k].update(v)

                    if missing:
                        validation_flags.append({
                            "type": "missing",
                            "chapter": i,
                            "group": group_idx,
                            "sentence_idx": sent_idx,
                            "entities": list(missing),
                        })

                    # -------------------------------
                    # 🔥 CRITICAL: SAVE STATE HERE
                    # -------------------------------
                    save_state(
                        global_entity_registry,
                        failed_sentences,
                        validation_flags,
                        i,
                        group_idx,
                        sent_idx
                    )

                    success = True
                    break

                except Exception as e:
                    print(f"[Retry {attempt+1}] Failed:", e)
                    time.sleep(2)

            if not success:
                failed_sentences.append({
                    "chapter": i,
                    "group": group_idx,
                    "sentence_idx": sent_idx
                })

                # save even on failure
                save_state(
                    global_entity_registry,
                    failed_sentences,
                    validation_flags,
                    i,
                    group_idx,
                    sent_idx
                )


print("\nProcessing complete. Final state saved in processing_state.json")
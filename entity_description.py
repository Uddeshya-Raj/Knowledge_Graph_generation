import json
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI

# Assume CLIENT is already defined (e.g., from openai import OpenAI)
# and call_llm function is available as shown in the prompt.

CLIENT = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="none")

def call_llm(system_prompt, user_query, model_name="Qwen/Qwen2.5-14B-Instruct", max_tokens=8192, response_model=None):
    api_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": max_tokens,
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

class EntityDescription(BaseModel):
    scratchpad: str = Field(
        description="Think how would you explain this is entity to someone who is seeing this entity for the first time."
    )
    description: str = Field(
        description="Final general purpose description/definition/meaning of the entity"
    )

def main():
    # Load existing entity info and id mapping
    with open("json_files/entity_info.json", "r", encoding="utf-8") as f:
        entity_info = json.load(f)

    with open("json_files/entity_id_dict.json", "r", encoding="utf-8") as f:
        entity_to_id = json.load(f)  # not directly used, but kept for completeness

    processed_entities = set()

    # Optional: load existing descriptions to avoid reprocessing
    for ent, data in entity_info.items():
        if data.get("description"):
            processed_entities.add(ent)

    # System prompt for the LLM
    system_prompt = """
You are generating dictionary-style entity descriptions from a Hindi book.

Your task is to identify the GENERAL meaning, role, identity, or category of the entity — not the specific event, ritual instance, sentence context, or chapter-specific occurrence.

Write exactly ONE concise sentence in Hindi (Devanagari script).

Guidelines:
1. Describe what the entity fundamentally is.
2. Prefer broad, reusable definitions over context-specific descriptions.
3. If the entity is:
   - a festival/ritual -> describe the ritual generally
   - a person/deity -> describe who they are generally
   - a text/scripture -> describe its purpose or nature
   - a time/unit/date -> describe the calendrical concept generally
   - an object/substance -> describe what it is generally
   - an action/process -> describe the action generally
4. Use chapter context only to disambiguate meaning, not to copy chapter details.
5. Avoid mentioning:
   - specific incidents
   - one particular ritual occurrence
   - chapter events
   - temporary usage in a sentence
   - examples unless absolutely necessary
   - detailed explanations
   - lists of actions
6. Do NOT define an entity using another very specific sentence from the chapter.
7. Do NOT include phrases like:
   - "इस दिन..."

   unless the entity itself is specifically that event/date.
8. Keep descriptions self-contained and generally valid across contexts.

Good examples:
- "विधान धार्मिक कार्यों को सम्पन्न करने की निर्धारित प्रक्रिया या नियम है।"
- "पञ्चाङ्ग हिन्दू कालगणना और ज्योतिषीय गणना का ग्रन्थ है।"
- "नीम एक औषधीय गुणों वाला वृक्ष है।"

Bad examples:
- "विधान में नीम के पत्ते खाना और पूजा करना शामिल है।"
- "नीम संवत्सरारम्भ के दिन खाया जाता है।"
- "पञ्चाङ्ग-श्रवण वह पर्व है जिसमें इस दिन पञ्चाङ्ग सुना जाता है।"

Return ONLY valid JSON:
{
    "scratchpad": "<identificaiton of the properties of the entities that describe it>",
    "description": "<single Hindi sentence>"
}
"""

    for chapter_idx in range(10):
        # Read chapter text
        with open(f"text_files/chapter_{chapter_idx}.txt", "r", encoding="utf-8") as f:
            chapter_text = f.read()

        # Load entity extraction outputs
        with open(f"entity_output_files/chapter_{chapter_idx}_output.json", "r", encoding="utf-8") as f:
            old_file = json.load(f)
        with open(f"entity_output_files_redone/chapter_{chapter_idx}_output.json", "r", encoding="utf-8") as f:
            new_file = json.load(f)

        # Build mapping: entity -> set of sentences (only from this chapter)
        chapter_entity_sentences = defaultdict(set)

        for file in (old_file, new_file):
            for group in file:
                for obj in group["extracted_entities"]:
                    sentence = obj["sentence"]
                    for entity in obj["entities"]:
                        chapter_entity_sentences[entity].add(sentence)

        # Generate description for each unseen entity in this chapter
        for entity, sentences in chapter_entity_sentences.items():
            if entity in processed_entities:
                continue
            if entity not in entity_to_id:
                print(f"Warning: Entity '{entity}' not found in entity_id_dict.json. Skipping.")
                continue

            # Prepare user query
            sentences_text = "\n".join(sentences)
            entity_id = entity_to_id[entity]
            user_query = (
                f"Entity: {entity}\n\n"
                f"Chapter text:\n{chapter_text}\n\n"
                # f"Sentences where '{entity}' appears:\n{sentences_text}\n\n"
                f"Generate a one-sentence description which identifies what that entity is or means."
            )

            try:
                response = call_llm(
                    system_prompt=system_prompt,
                    user_query=user_query,
                    model_name="Qwen/Qwen2.5-14B-Instruct",
                    max_tokens=512,          # description is short
                    response_model=EntityDescription
                )
                # response is a JSON string, parse it
                parsed = json.loads(response)
                description = parsed.get("description", "")
                if description:
                    entity_info[entity_id]["description"] = description
                    processed_entities.add(entity)
                    print(f"Generated description for '{entity}': {description}")
                else:
                    print(f"Empty description returned for '{entity}'")
            except Exception as e:
                print(f"Error processing entity '{entity}' in chapter {chapter_idx}: {e}")

        # Save after every chapter to avoid losing progress
        with open("json_files/entity_info.json", "w", encoding="utf-8") as f:
            json.dump(entity_info, f, ensure_ascii=False, indent=2)

    print("All descriptions generated and saved.")

if __name__ == "__main__":
    main()
    
    
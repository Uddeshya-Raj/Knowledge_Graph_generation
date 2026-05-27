import fasttext
import numpy as np
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# Global variables for lazy loading models
_FASTTEXT_MODEL = None
_TOKENIZER = None
_EMBED_MODEL = None
_DEVICE = None

def _initialize_models(load_fasttext=True, fasttext_path='model/indiclid-ftn/model_baseline_native.bin'):
    """Loads models into memory only once when needed."""
    global _FASTTEXT_MODEL, _TOKENIZER, _EMBED_MODEL, _DEVICE
    
    if load_fasttext and _FASTTEXT_MODEL is None:
        print("Loading FastText Model...")
        _FASTTEXT_MODEL = fasttext.load_model(fasttext_path)
        
    if _EMBED_MODEL is None:
        print("Loading BGE-M3 Model...")
        model_name = "BAAI/bge-m3"
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _EMBED_MODEL = AutoModel.from_pretrained(model_name)
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _EMBED_MODEL.to(_DEVICE)
        _EMBED_MODEL.eval()

def _check_lang(text):
    labels, probs = _FASTTEXT_MODEL.predict(text, k=1)
    return labels[0].replace("__label__", ""), probs[0]

def _overlap_tag_splitter(full_text):
    # Matches the new tag format or Hindi punctuation marks
    # We use capturing groups so re.split keeps the delimiters in the list
    delimiter_pattern = r'(<tag id="[^"]*" content="[^"]*">|[।॥\n]+)'
    raw_parts = re.split(delimiter_pattern, full_text)
    
    final_sentences = []
    current_segment = ""

    for part in raw_parts:
        if not part: continue
        
        # TYPE 1: The Tag (The Pivot)
        if part.startswith("<tag"):
            # A: If we have text before this tag, close that sentence with the tag
            if current_segment.strip():
                final_sentences.append(current_segment + part)
            
            # B: Immediately start the NEXT sentence with the same tag
            current_segment = part
            
        # TYPE 2: Hindi Punctuation (The Hard Stop)
        elif re.match(r'[।॥\n]+', part):
            if current_segment.strip():
                final_sentences.append(current_segment + part)
            current_segment = ""
            
        # TYPE 3: Regular Hindi Text
        else:
            current_segment += part

    # Catch any remaining text at the end
    if current_segment.strip():
        final_sentences.append(current_segment)
        
    return final_sentences

def _get_bge_embeddings(sentences, batch_size=32):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        encoded = _TOKENIZER(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(_DEVICE)
        
        with torch.no_grad():
            outputs = _EMBED_MODEL(**encoded)
            # BGE-M3 uses the [CLS] token (index 0) for its dense representation
            embeddings = outputs.last_hidden_state[:, 0]
            # Unit Normalization (L2) - CRITICAL for BGE-M3
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0)

def _generate_semantic_chunks(
    embeddings, 
    original_sentences, 
    base_threshold, 
    threshold_step=0.01, 
    alpha_ewma=0.3, 
    lookahead_size=2, 
    max_sentences=60, 
    overlap=2
):
    if len(embeddings) == 0:
        return []

    chunks = []
    current_chunk_text = []
    current_chunk_embeddings = []
    running_avg_vec = None
    
    i = 0
    while i < len(embeddings):
        curr_vec = embeddings[i]
        curr_text = original_sentences[i]

        if running_avg_vec is None:
            running_avg_vec = curr_vec
            current_chunk_text.append(curr_text)
            current_chunk_embeddings.append(curr_vec)
            i += 1
            continue

        lookahead_end = min(i + lookahead_size, len(embeddings))
        lookahead_vecs = embeddings[i:lookahead_end]
        lookahead_avg = np.mean(lookahead_vecs, axis=0)

        current_threshold = base_threshold + (len(current_chunk_text) * threshold_step)
        current_threshold = min(current_threshold, 0.98)

        sim = cosine_similarity(
            running_avg_vec.reshape(1, -1), 
            lookahead_avg.reshape(1, -1)
        )[0][0]

        # Condition to split: Semantic shift OR Hard length limit reached
        if sim < current_threshold or len(current_chunk_text) >= max_sentences:
            
            # 1. Save the completed chunk
            chunks.append(" ".join(current_chunk_text))
            
            # 2. Extract overlap sentences to bridge context into the new chunk
            if len(current_chunk_text) > overlap:
                overlap_texts = current_chunk_text[-overlap:]
                overlap_vecs = current_chunk_embeddings[-overlap:]
            else:
                overlap_texts = []
                overlap_vecs = []
                
            # 3. Start the next chunk with the overlap + the current sentence
            current_chunk_text = overlap_texts + [curr_text]
            current_chunk_embeddings = overlap_vecs + [curr_vec]
            
            # 4. Recalculate centroid based on the overlap sentences and current sentence
            running_avg_vec = np.mean(current_chunk_embeddings, axis=0)
            
            i += 1
        else:
            # Continue the current chunk
            running_avg_vec = (1 - alpha_ewma) * running_avg_vec + alpha_ewma * curr_vec
            current_chunk_text.append(curr_text)
            current_chunk_embeddings.append(curr_vec)
            i += 1

    # Catch the remaining trailing sentences
    if current_chunk_text:
        chunks.append(" ".join(current_chunk_text))

    return chunks

def chunk_text_by_alpha(
    text: str, 
    alpha: Optional[float] = None, 
    detect_language: bool = True, 
    max_sentences: int = 60,
    overlap: int = 2,
    fasttext_path: str = 'model/indiclid-ftn/model_baseline_native.bin'
):
    """
    Processes the text, calculates dynamic threshold, and returns semantic chunks.
    
    Args:
        text (str): The raw Hindi/Sanskrit text.
        alpha (float, optional): A value between 0.0 and 1.0. If None, uses avg cosine similarity.
        detect_language (bool): If True, isolates Sanskrit text. If False, bypasses language detection.
        max_sentences (int): Hard limit on sentences per chunk to prevent LLM context overflow.
        overlap (int): Number of sentences to duplicate across chunk boundaries to preserve context.
        fasttext_path (str): Path to the IndicLID fasttext model.
        
    Returns:
        dict: A dictionary containing the final chunks, extracted Sanskrit mapping, and metadata.
    """
    if alpha is not None and not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    _initialize_models(load_fasttext=detect_language, fasttext_path=fasttext_path)

    # 1. Split Text & Identify Sanskrit (if enabled)
    parts = re.split(r'([।:॥\-]+)', text)
    sentences = [
        parts[i].strip() + parts[i + 1]
        for i in range(0, len(parts) - 1, 2)
        if parts[i].strip()
    ]
    
    sans_dict = {}
    counter = 0
    full_text = ""
    
    for sentence in sentences:
        if '\n' in sentence:
            sentence = sentence.replace('\n', ' ')
            
        clean_sent = sentence.strip()
        
        if detect_language:
            lang, _ = _check_lang(clean_sent)
            if lang in ['san_Deva', 'san_Latn']:
                tag_id = f"doc_{counter}"
                sans_dict[tag_id] = clean_sent
                
                # Protect against double quotes in text breaking the tag attribute
                safe_content = clean_sent.replace('"', "'")
                full_text += f'<tag id="{tag_id}" content="{safe_content}">'
                counter += 1
            else:
                full_text += clean_sent + " "
        else:
            # Bypass language detection entirely
            full_text += clean_sent + " "

    # 2. Re-split with overlap logic
    overlap_sentences = _overlap_tag_splitter(full_text)
    
    # Remove the tag for model comprehension (leaves only Hindi context)
    clean_sentences = [re.sub(r'<tag id="[^"]*" content="[^"]*">', "", s).strip() for s in overlap_sentences]
    
    valid_indices = [i for i, s in enumerate(clean_sentences) if len(s) > 0]
    clean_for_model = [clean_sentences[i] for i in valid_indices]
    original_for_db = [overlap_sentences[i] for i in valid_indices]

    if not clean_for_model:
        return {"chunks": [], "sanskrit_mapping": sans_dict}

    # 3. Generate Embeddings
    embeddings = _get_bge_embeddings(clean_for_model, batch_size=32)

    # 4. Calculate dynamic threshold
    if len(embeddings) > 1:
        # Optimized adjacent similarity calculation using dot product on L2 normalized vectors
        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        min_a = float(np.min(sims))
        max_a = float(np.max(sims))
        avg_a = float(np.mean(sims))
    else:
        min_a, max_a, avg_a = 0.0, 1.0, 0.5

    # Determine Threshold Strategy
    if alpha is not None:
        dynamic_threshold = alpha * (max_a - min_a) + min_a
        threshold_source = f"Alpha={alpha}"
    else:
        dynamic_threshold = avg_a
        threshold_source = "Average"
        
    print(f"Calculated Threshold ({threshold_source}): {dynamic_threshold:.4f} (Min: {min_a:.4f}, Max: {max_a:.4f}, Avg: {avg_a:.4f})")

    # 5. Semantic Chunking with Limit & Overlap
    final_chunks = _generate_semantic_chunks(
        embeddings=embeddings,
        original_sentences=original_for_db,
        base_threshold=dynamic_threshold,
        threshold_step=0.005,
        alpha_ewma=0.3,
        lookahead_size=2,
        max_sentences=max_sentences,
        overlap=overlap
    )

    return {
        "chunks": final_chunks,
        "sanskrit_mapping": sans_dict,
        "metrics": {
            "min_sim": min_a,
            "max_sim": max_a,
            "avg_sim": avg_a,
            "applied_threshold": dynamic_threshold,
            "threshold_strategy": threshold_source,
            "max_sentences_cap": max_sentences,
            "overlap_size": overlap
        }
    }
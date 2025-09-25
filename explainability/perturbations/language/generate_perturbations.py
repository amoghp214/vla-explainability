import csv
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# Load LLM (T5 Paraphraser)
# ----------------------------
MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_t5(text: str, num_return_sequences: int = 5) -> list[str]:
    """Generate multiple unique paraphrases using T5 model with controlled sampling."""
    prompt = "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(
        prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )
    input_ids, attention_masks = encoding["input_ids"].to(model.device), encoding["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=num_return_sequences,
    )

    sentences = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sentences.append(line)

    # Deduplicate while preserving order
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    return unique_sentences[:num_return_sequences]

# ----------------------------
# Character-Level Perturbations
# ----------------------------
def keyboard_typo(text: str) -> str:
    if not text:
        return text
    i = random.randint(0, len(text) - 1)
    return text[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[i+1:]

def ocr_error(text: str) -> str:
    ocr_map = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "@"}
    return "".join(ocr_map.get(c, c) for c in text)

def concat_insertion(text: str) -> str:
    return text.replace(" ", "")

def char_replacement(text: str) -> str:
    if not text:
        return text
    i = random.randint(0, len(text) - 1)
    return text[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[i+1:]

def char_swap(text: str) -> str:
    if len(text) < 2:
        return text
    i = random.randint(0, len(text) - 2)
    return text[:i] + text[i+1] + text[i] + text[i+2:]

def char_delete(text: str) -> str:
    if not text:
        return text
    i = random.randint(0, len(text) - 1)
    return text[:i] + text[i+1:]

# ----------------------------
# Word-Level Perturbations
# ----------------------------
def word_swap(text: str) -> str:
    words = text.split()
    if len(words) > 1:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

def word_delete(text: str) -> str:
    words = text.split()
    if words:
        words.pop(random.randrange(len(words)))
    return " ".join(words)

def insert_punctuation(text: str) -> str:
    words = text.split()
    if not words:
        return text
    idx = random.randrange(len(words))
    words[idx] += random.choice(["?", "!", ",", "."])
    return " ".join(words)

# ----------------------------
# Sentence-Level Perturbations (Only 5 unique paraphrases)
# ----------------------------
def paraphrases(text: str) -> dict:
    generated = generate_t5(text, num_return_sequences=5)
    return {f"paraphrase{i}": p for i, p in enumerate(generated)}

# ----------------------------
# Main Pipeline
# ----------------------------
def generate_perturbations(text: str) -> dict:
    perturbations = {
        "keyboard": keyboard_typo(text),
        "ocr": ocr_error(text),
        "ci": concat_insertion(text),
        "cr": char_replacement(text),
        "cs": char_swap(text),
        "cd": char_delete(text),
        "ws": word_swap(text),
        "wd": word_delete(text),
        "ip": insert_punctuation(text),
    }
    perturbations.update(paraphrases(text))
    return perturbations

def save_to_csv(original_text: str, perturbations: dict, filename="perturbations.csv"):
    headers = ["original"] + list(perturbations.keys())
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        row = [original_text] + list(perturbations.values())
        writer.writerow(row)

if __name__ == "__main__":
    sentence = "take the broccoli out of the pan"
    perturbations = generate_perturbations(sentence)
    save_to_csv(sentence, perturbations)
    print("Perturbations saved to perturbations.csv")
    for k, v in perturbations.items():
        print(f"{k}: {v}")

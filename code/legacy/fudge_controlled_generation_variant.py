import torch
import pandas as pd
from pathlib import Path

import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizer,
    BertForSequenceClassification
)

#model_id = "meta-llama/Llama-3.2-3B-Instruct" #can use bigger/newer models here/2
#model_id = "meta-llama/Llama-3.2-1B-Instruct" #can use bigger/newer models here/1
#model_id = "meta-llama/Llama-3.1-8B-Instruct" #/4
model_id = "meta-llama/Meta-Llama-3-8B-Instruct" #/3
#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOP_K_VALUES = [50, 500]  #rescore 50 highest-probl tokens
max_tokens=150 #max output is 150 tokens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# LOAD YOUR TRAINED BERT MODELS (controllers)
# ======================================================

CLICKBAIT_MODEL_DIR = "bert_clickbait_prefix_finetuned"
TACTICS_MODEL_DIR   = "bert_tactics_prefix_finetuned"

clickbait_tokenizer = BertTokenizer.from_pretrained(CLICKBAIT_MODEL_DIR)
clickbait_model = BertForSequenceClassification.from_pretrained(
    CLICKBAIT_MODEL_DIR
).to(DEVICE).eval()

tactics_tokenizer = BertTokenizer.from_pretrained(TACTICS_MODEL_DIR)
tactics_model = BertForSequenceClassification.from_pretrained(
    TACTICS_MODEL_DIR
).to(DEVICE).eval()




# ======================================================
# SCORING FUNCTIONS (controllers)
# ======================================================

@torch.no_grad()
def score_clickbait_logit(text: str) -> float:
    enc = clickbait_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=32
    ).to(DEVICE)

    logits = clickbait_model(**enc).logits
    return logits[0, 1].item()


@torch.no_grad()
def score_tactics(text: str, weights):
    enc = tactics_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=32
    ).to(DEVICE)

    logits = tactics_model(**enc).logits
    probs = torch.sigmoid(logits)[0]   # shape: [num_tactics]

    weights_t = torch.tensor(weights, device=DEVICE)

    positive = weights_t > 0
    if positive.sum().item() == 0:
        return 0.0

    # average only the selected tactics -> stays in [0,1]
    return probs[positive].mean().item()



TACTIC_NAMES = [
    "curiosity gap", #kacha kacha
    "exaggeration", #good
    "emotional trigger", #hard to detect
    "sensationalism", #kacha kacha
    "lists or superlatives", #good
    "ambiguous references", # hard to detect
    "direct appeals",
    "unfinished narratives",
    "unexpected associations",
    "provocative questions" #good
]

ALPHAS = [0.0,1.0,2.0]
BETAS  = [0.0,1.0,2.0]

TACTIC_CONFIGS = [

    #([0], "Curiosity Gap"),
    #([1], "Exaggeration"),
    #([2], "Emotional Trigger"),
    #([3], "Sensationalism"),
    #([4], "Lists Or Superlatives"),
    #([5], "Ambiguous References"),
    #([6], "Direct Appeals"),
    #([7], "Unfinished Narratives"),
    #([8], "Unexpected Associations"),
    #([9], "Provocative Questions"),
    ([0, 3], "Curiosity Gap + sensationalism"),
    ([0, 2], "Curiosity Gap + Emotional Trigger"),
    ([0, 9], "Curiosity Gap + Provocative Questions"),
    ([2, 3], "Emotional Trigger + Sensationalism"),
    ([3, 9], "Sensationalism + Question"),



]

TACTIC_WEIGHTS = [1,0,0,0,0,0,0,0,0,0]  # example: curiosity only
DATA_PATH = Path(r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\combined_news_test.csv")

df = pd.read_csv(DATA_PATH)



# =====================================================
# 1) CHECK CAPITAL OR LOWER CASE LATTERS
# ====================================================
print("\n===== CAPITAL LETTER TEST =====")

test_prefixes = [
    "what remains hidden",
    "What remains hidden",
    "WHAT REMAINS HIDDEN"
]

for text in test_prefixes:
    cb = score_clickbait_logit(text)
    tac = score_tactics(text, TACTIC_WEIGHTS)



    print(f"\nInput: {text}")
    print(f"Clickbait logit: {cb:.4f}")
    print(f"Tactics score  : {tac:.4f}")


# ============================================


# =====================================================
# 2) CHECK EOS DOT EFFECT ON SCORING
# =====================================================

print("\n===== EOS DOT TEST =====")

test_sentence = "What's Behind the Hurdle to Trump's NAFTA Withdrawal"

cb_no_dot = score_clickbait_logit(test_sentence)
cb_dot    = score_clickbait_logit(test_sentence + ".")

tac_no_dot = score_tactics(test_sentence, TACTIC_WEIGHTS)
tac_dot    = score_tactics(test_sentence + ".", TACTIC_WEIGHTS)

print("\nWITHOUT dot:")
print("Clickbait:", cb_no_dot)
print("Tactics  :", tac_no_dot)

print("\nWITH dot:")
print("Clickbait:", cb_dot)
print("Tactics  :", tac_dot)


# ================================================================






def create_prompt(neutral_headline, tactics):

#   Rewrite the following neutral news headline to improve reader engagement using the tactics: {tactics}.
#   Rewrite the headline in a style that reflects: {tactics}.
  return f"""
  
  Rewrite the following neutral news headline.
  The wording may be lightly influenced by the following style: {tactics}.
  The influence should be subtle.

IMPORTANT:
- Do NOT mention the name of the tactic in the output.
- The tactic must be expressed only through wording style and tone.

STRICT RULES:
- Do NOT add any new facts
- Do NOT add any new people, places, or events
- Do NOT change the meaning
- Do NOT introduce claims not present in the original headline
- Only rephrase the wording

Output:
One headline only. No explanations.

Neutral headline:

{neutral_headline}
  """



#recieves current prefix(with canddiate token added) and canddiate token score
#should reduren combined score: exp(last_logit)+alpha*tactics_score(prefix, tactic_weights) -beta* clickbait_score(prefix)
#be careful with scales, either all are probabilities(so logit needs to be exponentiated) or all are probabilities
def score_prefix(prefix, next_token_logits, tok_id, alpha, beta,step):


    # alpha grows with sentence length
    alpha_dynamic = alpha * min(step / 6, 1.0)

    # 1) LLM term — log-probability (CLEAN, same scale)
    p_llm = torch.softmax(next_token_logits, dim=-1)[tok_id].item()



    # 2) Clickbait penalty — probability
    cb_logit = score_clickbait_logit(prefix)
    p_cb = torch.sigmoid(torch.tensor(cb_logit)).item()

    # 3) Tactic reward — probability
    tac = score_tactics(prefix, TACTIC_WEIGHTS)


# =============================================================
    # DEBUG: print tactic score for early tokens
    #if step < 5:
        #print("TACTIC SCORE:", tac)
# =============================================================


    # 4) Combined objective (all scales consistent)
    #combined_score = p_llm + alpha * tac - beta * p_cb
    combined_score = p_llm + alpha_dynamic * tac - beta * p_cb

    return combined_score

def get_next_token_id(generated_ids, next_token_logits, top_k_indices, alpha, beta):

    #min_length = 20
    best_score = -float("inf")
    best_token_id = None

    for tok_id in top_k_indices:

        #if tok_id == tokenizer.eos_token_id and len(generated_ids) < min_length:
            #continue

        candidate_text = tokenizer.decode(
            generated_ids + [tok_id],
            clean_up_tokenization_spaces=False
        )

        # ======================================================
        #CHECK EOS TOKEN
        # ======================================================

        if tok_id == tokenizer.eos_token_id:
            candidate_text = candidate_text.rstrip()
            if not candidate_text.endswith("."):
                candidate_text += "."

        # ======================================================

        # =====================================================

            # ---------------------------------
            # LENGTH CONTROL
            # ---------------------------------
            length = len(candidate_text.split())

            length_bonus = 0.0

            if length < 6:
                length_bonus = -0.4
            elif 8 <= length <= 30:
                length_bonus = +0.25
            elif length > 32:
                length_bonus = -0.2

            # ======================================================

        score = score_prefix(
            prefix=candidate_text,
            next_token_logits=next_token_logits,
            tok_id=tok_id,
            alpha=alpha,
            beta=beta,
            step = len(generated_ids)
        )

        if tok_id == tokenizer.eos_token_id:
            score += length_bonus


# ==================================================================
        #  DEBUG
        #if alpha > 0 and len(generated_ids) < 5:
            #token_str = tokenizer.decode([tok_id])
            #print(f"TOKEN: {token_str} | SCORE: {score:.3f}")
# ===================================================================

        if score > best_score:
            best_score = score
            best_token_id = tok_id

    return best_token_id



# ======================================
# LOAD LLM
# ======================================
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.eval()


results = []

results = []


for idx, row in df.iloc[0:5].iterrows():
#for idx, row in df.head(5).iterrows():

    neutral_headline = row["title"]

    for tactic_ids, tactic_label in TACTIC_CONFIGS:

        TACTIC_WEIGHTS = [1 if i in tactic_ids else 0 for i in range(len(TACTIC_NAMES))]
        tactic_names = [TACTIC_NAMES[i] for i in tactic_ids]

        print("FULL HEADLINE TACTIC SCORE:",
              score_tactics(neutral_headline, TACTIC_WEIGHTS))

        for top_k in TOP_K_VALUES:
            for alpha in ALPHAS:
                for beta in BETAS:

                    prompt = create_prompt(neutral_headline, tactic_names)

                    messages = [
                        {"role": "system", "content": "You are an experienced news editor."},
                        {"role": "user", "content": prompt}
                    ]

                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)

                    generated_ids = []

                    with torch.no_grad():
                        for _ in range(max_tokens):

                            outputs = model(input_ids=input_ids)
                            next_token_logits = outputs.logits[:, -1, :]

                            top_k_logits, top_k_indices = torch.topk(
                                next_token_logits, k=top_k, dim=-1
                            )

                            next_token_id = get_next_token_id(
                                generated_ids,
                                next_token_logits[0],
                                top_k_indices.tolist()[0],
                                alpha,
                                beta
                            )

                            if next_token_id == tokenizer.eos_token_id:
                                print("EOS chosen at step:", len(generated_ids))
                                break

                            generated_ids.append(next_token_id)

                            input_ids = torch.cat(
                                [input_ids, torch.tensor([[next_token_id]], device=model.device)],
                                dim=-1
                            )

                    # ===== decode AFTER generation loop =====
                    generated_text = tokenizer.decode(generated_ids)
                    generated_text = generated_text.strip()

                    if not generated_text.endswith((".", "?", "!")):
                        generated_text += "."

                    print("\n" + "=" * 80)
                    print(f"NEUTRAL : {neutral_headline}")
                    print(f"TACTICS : {tactic_label}")
                    print(f"TOP-K   : {top_k}")
                    print(f"ALPHA   : {alpha}")
                    print(f"BETA    : {beta}")
                    print(f"EDITED  : {generated_text}")
                    print("=" * 80)

                    results.append({
                        "neutral": neutral_headline,
                        "tactics": tactic_label,
                        "alpha": alpha,
                        "beta": beta,
                        "edited": generated_text,
                        "top_k": top_k
                    })


results_df = pd.DataFrame(results)

results_df.to_csv(
    "prefix_generation_results_3.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved results to prefix_generation_results_3.csv")


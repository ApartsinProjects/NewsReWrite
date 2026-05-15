"""One-off hero image generator for the NewsReWrite README."""
import json
from pathlib import Path

from google import genai
from google.genai import types

CONFIG = json.loads(Path.home().joinpath(".gemini-imagegen.json").read_text())
OUTPUT = Path(__file__).parent / "hero.png"

PROMPT = """
A cinematic editorial-research hero banner for a NLP project named
"NewsReWrite: LLM-Guided Headline Rewriting".

Composition:
- Center: a stylized newspaper sheet being rewritten in mid-air, fragmenting
  into a stream of glowing tokens that flow horizontally through a pipeline.
- Two opposing fields converge on the token stream:
    * From above, a warm amber-and-teal "positive guidance" beam, depicted as
      soft glowing rays representing engagement attributes (curiosity,
      emphasis, narrative pull).
    * From below, a cool deep-blue "negative guidance" suppression field,
      pushing back against clickbait amplification, depicted as cold ribbons
      of suppressed energy.
- The two fields meet on a horizontal continuous spectrum slider rendered
  as a luminous rail; a small marker sits at a balanced, responsible point,
  not at the extreme.
- Background: faint, blurred mathematical decoding equations and
  transformer attention nodes (small connected glowing dots forming a
  neural graph), like a research paper's diagram dissolving into atmosphere.
- Subtle BERT-like encoder block silhouettes flanking the pipeline.
- Particles of light, depth of field, slight glassmorphism on overlay panels.

Style:
- Editorial, research-paper aesthetic, premium tech publication look.
- Palette: dark navy and near-black background, warm amber highlights,
  teal/cyan accents, restrained use of cool blue for the suppression field.
- Cinematic volumetric lighting, high detail, crisp focal subject,
  soft bokeh background.
- 16:9 hero banner suitable for the top of a GitHub README.
- Absolutely no readable text, letters, numbers, words, captions or
  watermarks anywhere in the image.
"""

client = genai.Client(api_key=CONFIG["api_key"])

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=PROMPT,
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="16:9", image_size="2K"),
    ),
)

saved = False
for part in response.parts:
    if part.inline_data:
        part.as_image().save(OUTPUT)
        saved = True
        print(f"Saved: {OUTPUT}")
        break

if not saved:
    raise SystemExit("No image returned by the model")

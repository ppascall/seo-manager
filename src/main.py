import csv
import os
import sys
import time
import json
import re
import random
from difflib import SequenceMatcher
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

# Load environment variables from .env file
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
API_KEY = os.getenv("CEREBRAS_API_KEY", "")
MODEL_NAME = "llama-3.3-70b"

INPUT_CSV = "csv_input/product_export.csv"
OUTPUT_CSV = "csv_output/product_export_seo.csv"
PROGRESS_FILE = "csv_output/progress.json"

# Rate limiting: pause between API calls (seconds)
# Cerebras free tier: 30 requests/min, enforced as ~1 req/2s
API_DELAY = 5.0

# Similarity threshold: if a new description is more similar than this
# to ANY existing description, it gets regenerated (0.0 = totally different, 1.0 = identical)
SIMILARITY_THRESHOLD = 0.80
MAX_RETRIES = 3

# Shopify SEO limits
MAX_SEO_TITLE_LENGTH = 70
MAX_SEO_DESCRIPTION_LENGTH = 320

# Detected product type (set at runtime)
DETECTED_PRODUCT_TYPE = None  # "wall_clocks", "water_bottles", or "lamp_shades"

# ============================================================
# PRODUCT TYPE DETECTION KEYWORDS
# ============================================================
PRODUCT_TYPE_KEYWORDS = {
    "wall_clocks": [
        "clock", "wall clock", "timepiece", "hour hand", "minute hand",
        "quartz", "pendulum", "analog", "dial", "roman numeral",
        "sweep movement", "ticking", "clock face", "clock hands",
    ],
    "water_bottles": [
        "water bottle", "bottle", "tumbler", "hydration", "flask",
        "drink bottle", "sports bottle", "insulated bottle", "bpa free",
        "stainless steel bottle", "reusable bottle", "thermos",
        "beverage container", "sippy", "straw lid",
    ],
    "lamp_shades": [
        "lamp shade", "lampshade", "shade", "lamp", "light shade",
        "lighting", "pendant shade", "table lamp", "floor lamp",
        "drum shade", "cone shade", "bell shade", "fabric shade",
        "linen shade", "silk shade",
    ],
}


def detect_product_type(rows):
    """Scan product titles and descriptions to determine product type."""
    scores = {"wall_clocks": 0, "water_bottles": 0, "lamp_shades": 0}

    for row in rows:
        text = (
            row.get("Title", "") + " " +
            row.get("Body (HTML)", "") + " " +
            row.get("Type", "") + " " +
            row.get("Tags", "")
        ).lower()

        for product_type, keywords in PRODUCT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    scores[product_type] += 1

    detected = max(scores, key=scores.get)
    if scores[detected] == 0:
        print("WARNING: Could not confidently detect product type. Defaulting to wall_clocks.")
        return "wall_clocks"

    labels = {
        "wall_clocks": "Wall Clocks",
        "water_bottles": "Water Bottles",
        "lamp_shades": "Lamp Shades",
    }
    print(f"Detected product type: {labels[detected]} (score: {scores[detected]})")
    print(f"  All scores: {', '.join(f'{labels[k]}={v}' for k, v in scores.items())}")
    return detected


# ============================================================
# VOCABULARY POOLS PER PRODUCT TYPE
# ============================================================
VOCAB_POOLS = {
    # ---- WALL CLOCKS ----
    "wall_clocks": {
        "opening_angles": [
            "Describe the clock focusing on its design style and how it complements room decor.",
            "Describe the clock focusing on its materials and build quality.",
            "Describe the clock focusing on its movement type and timekeeping reliability.",
            "Describe the clock focusing on where it looks best — living rooms, offices, kitchens.",
            "Describe the clock focusing on its size and how it anchors a wall.",
            "Describe the clock focusing on its readability and dial design.",
            "Describe the clock as a statement piece for interior design.",
            "Describe the clock emphasizing gift-worthiness and presentation.",
            "Describe the clock from the perspective of a home decorator.",
            "Describe the clock focusing on easy installation and hanging.",
            "Describe the clock highlighting its finish and frame detailing.",
            "Describe the clock as a practical yet decorative home accent.",
        ],
        "settings": [
            "living rooms", "home offices", "kitchens", "bedrooms", "entryways",
            "dining rooms", "hallways", "lobbies", "waiting rooms", "classrooms",
            "cafés", "boutique shops", "hotel rooms", "loft apartments", "studios",
            "conference rooms", "coworking spaces", "libraries", "nurseries", "dens",
            "farmhouse kitchens", "gallery walls", "accent walls", "mantels", "above-fireplace spots",
        ],
        "materials": [
            "solid wood frame", "metal housing", "brushed aluminium case", "moulded plastic body",
            "tempered glass lens", "MDF backing", "wrought iron frame", "bamboo surround",
            "distressed wood finish", "powder-coated steel", "polished chrome bezel",
            "reclaimed wood face", "ceramic dial plate", "resin composite body",
            "hand-painted wooden frame", "galvanized metal trim", "natural oak frame",
            "matte black steel case", "antique brass finish", "whitewashed pine frame",
        ],
        "qualities": [
            "silent sweep movement for noise-free rooms", "accurate quartz timekeeping",
            "built to last with durable construction", "precision-engineered movement",
            "designed for years of reliable use", "quality craftsmanship throughout",
            "fade-resistant printed dial", "vibration-resistant wall mount",
            "consistent timekeeping with minimal drift", "long-lasting battery life",
        ],
        "style_words": [
            "minimalist", "farmhouse", "industrial", "mid-century modern", "vintage",
            "rustic", "contemporary", "Scandinavian", "art deco", "retro",
            "bohemian", "coastal", "traditional", "modern geometric", "shabby chic",
            "oversized statement", "classic round", "roman numeral", "clean-line",
            "gallery-style", "sunburst", "skeleton", "pendulum-style", "schoolhouse",
        ],
        "features": [
            "silent sweep second hand", "large easy-to-read numerals", "built-in hanging hook",
            "protective glass lens cover", "non-ticking quartz movement", "glow-in-the-dark hands",
            "temperature and humidity display", "date window", "Roman numeral dial",
            "Arabic numeral markings", "open-face skeleton design", "integrated LED backlight",
            "dual time zone display", "pendulum mechanism", "battery-powered operation",
        ],
        "dimensions": [
            "large statement size ideal for open walls", "compact design for smaller rooms",
            "oversized dial visible from across the room", "standard 12-inch diameter",
            "slim profile sits flush against the wall", "lightweight for easy hanging",
        ],
    },
    # ---- WATER BOTTLES ----
    "water_bottles": {
        "opening_angles": [
            "Describe the bottle focusing on its insulation and temperature retention.",
            "Describe the bottle focusing on its materials and safety certifications.",
            "Describe the bottle focusing on its portability and on-the-go convenience.",
            "Describe the bottle focusing on its lid design and leak-proof features.",
            "Describe the bottle focusing on its capacity and hydration goals.",
            "Describe the bottle focusing on its durability for outdoor and sport use.",
            "Describe the bottle focusing on its eco-friendly reusable design.",
            "Describe the bottle as a gym and fitness essential.",
            "Describe the bottle from the perspective of a daily commuter.",
            "Describe the bottle emphasizing ease of cleaning and maintenance.",
            "Describe the bottle highlighting its finish and colour options.",
            "Describe the bottle as a practical gift for health-conscious people.",
        ],
        "settings": [
            "gyms", "offices", "hiking trails", "yoga studios", "school classrooms",
            "commuter bags", "bike rides", "camping trips", "road trips", "desks",
            "outdoor festivals", "beach days", "running routes", "crossfit boxes",
            "travel carry-ons", "meal prep stations", "sports sidelines", "picnics",
            "home workouts", "warehouse floors", "workshop benches", "playgrounds",
            "marathon events", "backpacking trips", "daily errands",
        ],
        "materials": [
            "18/8 stainless steel", "BPA-free Tritan plastic", "double-wall vacuum insulation",
            "food-grade silicone seal", "powder-coated exterior", "copper-lined insulation",
            "borosilicate glass body", "recycled plastic construction", "medical-grade stainless",
            "rubberised grip coating", "bamboo lid accent", "shatter-resistant polymer",
            "electro-polished interior", "non-toxic coating", "FDA-approved materials",
            "scratch-resistant finish", "sweat-proof outer wall", "ergonomic soft-touch grip",
        ],
        "qualities": [
            "keeps drinks cold for 24 hours", "keeps beverages hot for 12 hours",
            "built for daily abuse and repeated drops", "designed for active lifestyles",
            "leak-proof seal you can trust in any bag", "engineered for one-handed drinking",
            "condensation-free exterior", "dishwasher-safe construction",
            "odour-resistant interior lining", "rust-proof and stain-resistant",
        ],
        "style_words": [
            "sleek", "sporty", "minimalist", "matte finish", "gradient colour",
            "urban", "outdoor-ready", "slim-profile", "wide-mouth", "classic",
            "bold colour-pop", "earth-toned", "monochrome", "textured grip",
            "pastel", "metallic sheen", "frosted", "transparent", "ombré",
        ],
        "features": [
            "flip-top straw lid", "wide mouth for ice cubes", "carry loop handle",
            "one-click open mechanism", "built-in fruit infuser", "measurement markings",
            "removable strainer", "carabiner clip attachment", "collapsible design",
            "dual-lid system — sport cap and screw top", "integrated tea filter",
            "spout cover for hygiene", "non-slip base pad", "finger grip indentations",
            "time marker hydration tracker", "compatible with car cup holders",
        ],
        "dimensions": [
            "fits standard car cup holders", "compact enough for side bag pockets",
            "large capacity for all-day hydration", "slim profile for gym bag storage",
            "lightweight at under 300 grams", "tall design maximises volume without bulk",
        ],
    },
    # ---- LAMP SHADES ----
    "lamp_shades": {
        "opening_angles": [
            "Describe the lamp shade focusing on the quality of light it produces.",
            "Describe the lamp shade focusing on its fabric and texture.",
            "Describe the lamp shade focusing on how it transforms a room's ambiance.",
            "Describe the lamp shade focusing on its shape and silhouette.",
            "Describe the lamp shade focusing on colour and how it coordinates with decor.",
            "Describe the lamp shade focusing on its fit and compatibility with lamp bases.",
            "Describe the lamp shade focusing on its construction and lining.",
            "Describe the lamp shade as an interior design statement piece.",
            "Describe the lamp shade from the perspective of a home stager.",
            "Describe the lamp shade emphasizing easy installation and bulb compatibility.",
            "Describe the lamp shade highlighting its proportions and scale.",
            "Describe the lamp shade as a cost-effective room refresh accessory.",
        ],
        "settings": [
            "living rooms", "bedrooms", "reading nooks", "bedside tables", "home offices",
            "dining areas", "hotel rooms", "boutique lobbies", "nurseries", "guest rooms",
            "hallway console tables", "accent corners", "window seats", "study desks",
            "spa treatment rooms", "restaurant tables", "cocktail lounges", "dressing rooms",
            "libraries", "loft apartments", "cottage interiors", "farmhouse kitchens",
            "gallery spaces", "foyers", "cosy dens",
        ],
        "materials": [
            "linen fabric", "cotton drum shade", "silk shade panel", "textured burlap",
            "pleated polyester", "handmade paper", "woven rattan shell", "frosted glass",
            "brass-trimmed ring", "nickel-finished spider fitter", "UNO fitter ring",
            "polished chrome hardware", "self-trim fabric edge", "styrene backing",
            "PVC lining for shape retention", "natural jute wrapping", "velvet exterior",
            "recycled fabric blend", "organza overlay", "linen-look polyester",
        ],
        "qualities": [
            "casts a warm, even glow across the room", "eliminates harsh overhead glare",
            "diffuses light softly for comfortable ambiance", "built to retain shape over time",
            "designed for years of use without sagging", "professional-grade construction",
            "colour-fast fabric resists fading", "flame-retardant material for safety",
            "dust-resistant surface for easy upkeep", "maintains crisp edges wash after wash",
        ],
        "style_words": [
            "drum", "empire", "bell", "coolie", "rectangular",
            "tapered", "pleated", "scalloped", "conical", "cylindrical",
            "mid-century", "Scandinavian", "coastal", "bohemian", "art deco",
            "classic neutral", "bold accent", "textured weave", "translucent",
            "opaque", "two-tone", "monochrome", "patterned", "geometric print",
        ],
        "features": [
            "spider fitter for harp-style bases", "UNO fitter for socket-ring bases",
            "clip-on attachment for chandelier bulbs", "washer fitter with reducer ring",
            "compatible with E26/E27 standard bases", "suitable for LED and CFL bulbs",
            "removable diffuser panel", "reversible inside-out design",
            "adjustable tilt mechanism", "integrated reflector lining",
            "snap-on attachment system", "heat-resistant inner lining",
            "top and bottom trim detailing", "reinforced wire frame",
        ],
        "dimensions": [
            "standard size fits most table lamps", "oversized for floor lamp bases",
            "mini size ideal for chandelier arms", "proportioned for bedside lamps",
            "slim profile for narrow console tables", "wide diameter for maximum light spread",
        ],
    },
}


def get_variation_hint(product_type):
    """Pick a random writing angle and vocab suggestions for prompt variation."""
    pool = VOCAB_POOLS[product_type]

    angle = random.choice(pool["opening_angles"])
    settings = random.sample(pool["settings"], 4)
    materials = random.sample(pool["materials"], 4)
    qualities = random.sample(pool["qualities"], 2)
    style = random.sample(pool["style_words"], 3)
    features = random.sample(pool["features"], 3)
    dims = random.sample(pool["dimensions"], 2)

    return (
        f"WRITING ANGLE: {angle}\n"
        f"Consider using some of these words/phrases where they fit naturally (do NOT force them all in, pick 3-5 max):\n"
        f"- Settings: {', '.join(settings)}\n"
        f"- Materials: {', '.join(materials)}\n"
        f"- Qualities: {', '.join(qualities)}\n"
        f"- Style: {', '.join(style)}\n"
        f"- Features: {', '.join(features)}\n"
        f"- Dimensions: {', '.join(dims)}"
    )


def check_similarity(new_desc, existing_descriptions):
    """Check if new_desc is too similar to any existing description. Returns (is_ok, worst_score, worst_match)."""
    if not existing_descriptions:
        return True, 0.0, ""
    worst_score = 0.0
    worst_match = ""
    for existing in existing_descriptions:
        score = SequenceMatcher(None, new_desc.lower(), existing.lower()).ratio()
        if score > worst_score:
            worst_score = score
            worst_match = existing
    return worst_score < SIMILARITY_THRESHOLD, worst_score, worst_match


# ============================================================
# PROMPT TEMPLATES PER PRODUCT TYPE
# ============================================================
SEO_PROMPTS = {
    "wall_clocks": """You are an SEO copywriter for a brand that sells wall clocks — decorative, functional, and design-forward timepieces for homes, offices, and commercial spaces. You write like someone who genuinely understands interior design and home decor.

Your task: Generate an SEO Title and SEO Description for a Shopify product.

STRICT RULES — follow every single one:

SEO TITLE RULES:
1. The SEO Title MUST be the exact product title provided. Copy it exactly. Do NOT change, add, or rephrase any words.

SEO DESCRIPTION RULES:
2. Write 1-3 natural, descriptive sentences about the product. This is NOT a keyword list. Do NOT use comma-separated keywords.
3. Use the primary keyword (the product type, e.g. "wall clock", "decorative timepiece", "modern wall clock") once, naturally, in the first sentence.
4. Each product MUST have a unique primary modifier. Pick the most relevant one for this specific product:
   - Style (e.g. minimalist, farmhouse, industrial, mid-century modern)
   - Material or finish (e.g. solid wood frame, brushed metal, matte black)
   - Size (e.g. oversized 24-inch, compact 10-inch)
   - Setting (e.g. living rooms, offices, kitchens)
5. After the primary keyword, cover whichever of these are relevant and mentioned in the product info:
   - Materials (frame, dial, glass, movement type)
   - Movement details (silent sweep, quartz, non-ticking)
   - Readability (large numerals, Roman numerals, clean dial)
   - Installation (wall-mount, hanging hook, battery type)
   - Settings and rooms where it fits best
6. The description MUST be under 320 characters total.
7. The description MUST NOT contain any HTML tags, quotes, newlines, or special formatting.
8. Do NOT invent features or details not found in the product info. Only describe what is actually mentioned.
9. Do NOT include pricing, availability, or promotional language.
10. Write naturally and specifically about this exact product. Vary your sentence structure and word choice.
11. Do NOT start with the brand name. Start directly with what the product is.

{variation_hint}

EXAMPLE INPUT:
Product Title: Nordic Minimalist Silent Wall Clock 12 Inch
Product Description: Simple Scandinavian design wall clock with silent sweep movement. 12-inch diameter, wooden frame with clean white dial. Battery operated, easy wall mount.

EXAMPLE OUTPUT:
SEO Title: Nordic Minimalist Silent Wall Clock 12 Inch
SEO Description: Scandinavian-style wall clock with silent sweep movement and clean white dial in a natural wooden frame. The 12-inch diameter suits bedrooms, offices and living rooms without ticking noise. Battery operated with easy wall mount.

NOW GENERATE FOR THIS PRODUCT:
Product Title: {title}
Product Description: {body}

Respond with ONLY these two lines, nothing else:
SEO Title: <exact product title>
SEO Description: <1-3 descriptive sentences, under 320 characters>""",

    "water_bottles": """You are an SEO copywriter for a brand that sells water bottles — reusable, insulated, and purpose-built hydration products for active and everyday use. You write like someone who actually tests gear for fitness, commuting, and outdoor adventures.

Your task: Generate an SEO Title and SEO Description for a Shopify product.

STRICT RULES — follow every single one:

SEO TITLE RULES:
1. The SEO Title MUST be the exact product title provided. Copy it exactly. Do NOT change, add, or rephrase any words.

SEO DESCRIPTION RULES:
2. Write 1-3 natural, descriptive sentences about the product. This is NOT a keyword list. Do NOT use comma-separated keywords.
3. Use the primary keyword (the product type, e.g. "insulated water bottle", "stainless steel bottle", "reusable sports bottle") once, naturally, in the first sentence.
4. Each product MUST have a unique primary modifier. Pick the most relevant one for this specific product:
   - Material (e.g. 18/8 stainless steel, BPA-free Tritan, borosilicate glass)
   - Insulation (e.g. double-wall vacuum, copper-lined)
   - Capacity (e.g. 500ml, 750ml, 1 litre)
   - Use case (e.g. gym, hiking, office, commuting)
   - Lid type (e.g. flip-top straw, wide-mouth screw, sport cap)
5. After the primary keyword, cover whichever of these are relevant and mentioned in the product info:
   - Materials and safety certifications (BPA-free, food-grade)
   - Temperature retention (hours cold/hot)
   - Lid and drinking mechanism
   - Portability (cup-holder fit, carry loop, weight)
   - Cleaning and maintenance
   - Durability and drop resistance
6. The description MUST be under 320 characters total.
7. The description MUST NOT contain any HTML tags, quotes, newlines, or special formatting.
8. Do NOT invent features or details not found in the product info. Only describe what is actually mentioned.
9. Do NOT include pricing, availability, or promotional language.
10. Write naturally and specifically about this exact product. Vary your sentence structure and word choice.
11. Do NOT start with the brand name. Start directly with what the product is.

{variation_hint}

EXAMPLE INPUT:
Product Title: Arctic Pro Insulated Water Bottle 750ml Matte Black
Product Description: Double-wall vacuum insulated stainless steel water bottle. Keeps drinks cold 24 hours, hot 12 hours. BPA-free, powder-coated finish, wide mouth opening for ice cubes. Leak-proof screw cap with carry loop.

EXAMPLE OUTPUT:
SEO Title: Arctic Pro Insulated Water Bottle 750ml Matte Black
SEO Description: Double-wall vacuum insulated stainless steel water bottle that keeps drinks cold for 24 hours and hot for 12. Wide mouth fits ice cubes easily, and the leak-proof screw cap has a carry loop for on-the-go use. BPA-free with a durable powder-coated finish.

NOW GENERATE FOR THIS PRODUCT:
Product Title: {title}
Product Description: {body}

Respond with ONLY these two lines, nothing else:
SEO Title: <exact product title>
SEO Description: <1-3 descriptive sentences, under 320 characters>""",

    "lamp_shades": """You are an SEO copywriter for a brand that sells lamp shades — fabric, paper, and structured shades for table lamps, floor lamps, pendants, and chandeliers. You write like an interior designer who knows how lighting transforms a space.

Your task: Generate an SEO Title and SEO Description for a Shopify product.

STRICT RULES — follow every single one:

SEO TITLE RULES:
1. The SEO Title MUST be the exact product title provided. Copy it exactly. Do NOT change, add, or rephrase any words.

SEO DESCRIPTION RULES:
2. Write 1-3 natural, descriptive sentences about the product. This is NOT a keyword list. Do NOT use comma-separated keywords.
3. Use the primary keyword (the product type, e.g. "drum lamp shade", "linen table lamp shade", "fabric pendant shade") once, naturally, in the first sentence.
4. Each product MUST have a unique primary modifier. Pick the most relevant one for this specific product:
   - Shape (e.g. drum, empire, bell, coolie, tapered)
   - Material (e.g. linen, silk, cotton, burlap, rattan)
   - Colour or pattern (e.g. cream, navy, geometric print)
   - Fitter type (e.g. spider, UNO, clip-on, washer)
   - Lamp type (e.g. table lamp, floor lamp, pendant, chandelier)
5. After the primary keyword, cover whichever of these are relevant and mentioned in the product info:
   - Material and fabric weight
   - Light quality (warm glow, diffused, directional)
   - Fitter compatibility and installation
   - Dimensions and proportions
   - Room suitability
   - Construction (lining, frame, trim)
6. The description MUST be under 320 characters total.
7. The description MUST NOT contain any HTML tags, quotes, newlines, or special formatting.
8. Do NOT invent features or details not found in the product info. Only describe what is actually mentioned.
9. Do NOT include pricing, availability, or promotional language.
10. Write naturally and specifically about this exact product. Vary your sentence structure and word choice.
11. Do NOT start with the brand name. Start directly with what the product is.

{variation_hint}

EXAMPLE INPUT:
Product Title: Classic Linen Drum Shade Natural 14 Inch
Product Description: Natural linen drum lamp shade with spider fitter. 14-inch diameter, 10-inch height. White styrene lining for even light diffusion. Compatible with standard E26 harp-style bases.

EXAMPLE OUTPUT:
SEO Title: Classic Linen Drum Shade Natural 14 Inch
SEO Description: Natural linen drum lamp shade with a clean, modern silhouette and white styrene lining for warm, even light diffusion. The 14-inch diameter fits standard E26 harp-style bases via the included spider fitter. Ideal for living rooms and bedrooms.

NOW GENERATE FOR THIS PRODUCT:
Product Title: {title}
Product Description: {body}

Respond with ONLY these two lines, nothing else:
SEO Title: <exact product title>
SEO Description: <1-3 descriptive sentences, under 320 characters>""",
}

def strip_html(html_text):
    """Remove HTML tags and clean up text for the prompt."""
    if not html_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate very long descriptions to keep prompt focused
    if len(text) > 800:
        text = text[:800] + "..."
    return text


def load_progress():
    """Load set of already-processed handles to avoid duplicates."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(processed_handles):
    """Save the set of processed handles."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(processed_handles), f, indent=2)


def parse_seo_response(response_text, original_title):
    """Parse the model response to extract SEO Title and SEO Description."""
    lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]

    seo_title = original_title  # Default: use exact product title
    seo_description = ""

    for line in lines:
        if line.lower().startswith("seo title:"):
            seo_title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("seo description:"):
            seo_description = line.split(":", 1)[1].strip()

    # Safety: always enforce the original title
    seo_title = original_title

    # Clean up the description
    seo_description = re.sub(r"<[^>]+>", "", seo_description)  # strip any HTML
    seo_description = seo_description.replace('"', "").replace("'", "")
    seo_description = seo_description.replace("faux leather", "vinyl")
    seo_description = seo_description.replace("Faux leather", "Vinyl")
    seo_description = seo_description.replace("Faux Leather", "Vinyl")
    seo_description = re.sub(r"\s+", " ", seo_description).strip()

    # Enforce length limits
    if len(seo_title) > MAX_SEO_TITLE_LENGTH:
        seo_title = seo_title[:MAX_SEO_TITLE_LENGTH]
    if len(seo_description) > MAX_SEO_DESCRIPTION_LENGTH:
        seo_description = seo_description[:MAX_SEO_DESCRIPTION_LENGTH]

    return seo_title, seo_description


def generate_seo(client, title, body_html, existing_descriptions, product_type):
    """Call Cerebras API to generate SEO fields, with similarity checking and retries."""
    clean_body = strip_html(body_html)

    for attempt in range(MAX_RETRIES):
        variation_hint = get_variation_hint(product_type)
        prompt_template = SEO_PROMPTS[product_type]
        prompt = prompt_template.format(title=title, body=clean_body, variation_hint=variation_hint)

        # Retry loop for transient 503 errors
        for server_retry in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an SEO copywriter. Follow instructions exactly. Output only what is asked, nothing else."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0.5 + (attempt * 0.15),
                )
                break  # success
            except Exception as e:
                err = str(e)
                if "503" in err or "502" in err or "429" in err or "server" in err.lower() or "too_many" in err.lower():
                    wait = (server_retry + 1) * 8  # 8s, 16s, 24s, 32s, 40s
                    print(f"    Server/rate limit error, waiting {wait}s before retry ({server_retry + 1}/5)...")
                    time.sleep(wait)
                    if server_retry == 4:
                        raise  # give up after 5 server retries
                else:
                    raise

        seo_title, seo_description = parse_seo_response(response.choices[0].message.content, title)

        # Check similarity against all previously generated descriptions
        is_ok, score, similar_to = check_similarity(seo_description, existing_descriptions)

        if is_ok:
            if attempt > 0:
                print(f"    (took {attempt + 1} attempts to get unique description)")
            return seo_title, seo_description
        else:
            print(f"    Attempt {attempt + 1}: too similar ({score:.0%}) to existing desc, retrying...")
            time.sleep(API_DELAY)

    # If all retries exhausted, use the last one anyway
    print(f"    Warning: could not get below {SIMILARITY_THRESHOLD:.0%} similarity after {MAX_RETRIES} attempts, using best result.")
    return seo_title, seo_description


def main():
    global DETECTED_PRODUCT_TYPE

    # --- Check for --overwrite flag ---
    overwrite = "--overwrite" in sys.argv

    # --- Setup Cerebras ---
    if not API_KEY:
        print("ERROR: CEREBRAS_API_KEY not found. Set it in the .env file.")
        return

    client = Cerebras(api_key=API_KEY)

    # --- Load progress ---
    if overwrite:
        processed_handles = set()
        # Clear progress file
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        print("OVERWRITE MODE: Regenerating all SEO data from scratch.")
    else:
        processed_handles = load_progress()
    print(f"Already processed: {len(processed_handles)} products")

    # --- Check input file exists ---
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        print("Place your Shopify product export as csv_input/product_export.csv")
        return

    # --- Read input CSV ---
    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Total rows in CSV: {len(rows)}")

    # --- Detect product type ---
    DETECTED_PRODUCT_TYPE = detect_product_type(rows)
    print(f"Using SEO vocabulary and prompts for: {DETECTED_PRODUCT_TYPE}")
    print()

    # --- Identify product rows needing SEO ---
    products_to_process = []
    for i, row in enumerate(rows):
        title = row.get("Title", "").strip()
        handle = row.get("Handle", "").strip()
        existing_seo_desc = row.get("SEO Description", "").strip()

        if overwrite:
            # In overwrite mode, process all product rows with a title
            if title and handle not in processed_handles:
                products_to_process.append((i, row))
        else:
            # Normal mode: skip rows that already have SEO or were previously processed
            if title and not existing_seo_desc and handle not in processed_handles:
                products_to_process.append((i, row))

    print(f"Products needing SEO generation: {len(products_to_process)}")

    # --- Collect existing descriptions for similarity checking ---
    existing_descriptions = []
    for row in rows:
        desc = row.get("SEO Description", "").strip()
        if desc and not overwrite:
            existing_descriptions.append(desc)

    if not products_to_process:
        print("Nothing to process. All products already have SEO data.")
    else:
        # --- Generate SEO for each product ---
        for idx, (row_index, row) in enumerate(products_to_process):
            title = row["Title"].strip()
            body = row.get("Body (HTML)", "")
            handle = row.get("Handle", "").strip()

            print(f"\n[{idx + 1}/{len(products_to_process)}] Generating SEO for: {title}")

            try:
                seo_title, seo_description = generate_seo(client, title, body, existing_descriptions, DETECTED_PRODUCT_TYPE)

                # Update the row in place
                rows[row_index]["SEO Title"] = seo_title
                rows[row_index]["SEO Description"] = seo_description

                # Add to similarity pool so next products are checked against this one
                existing_descriptions.append(seo_description)

                # Track progress
                processed_handles.add(handle)
                save_progress(processed_handles)

                print(f"  SEO Title: {seo_title}")
                print(f"  SEO Desc:  {seo_description}")

            except Exception as e:
                print(f"  ERROR: {e}")
                print("  Skipping this product, will retry on next run.")

            # Rate limiting
            if idx < len(products_to_process) - 1:
                time.sleep(API_DELAY)

    # --- Write output CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutput written to: {OUTPUT_CSV}")
    print("Done!")


if __name__ == "__main__":
    main()

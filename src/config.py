"""Single source of truth for classes, chunkers, paths and pipeline constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEOS = DATA / "videos"
FRAMES = DATA / "frames"
SHEETS = DATA / "sheets"
ART = ROOT / "artifacts"
FEATURES = ART / "features"
MODELS = ART / "models"
SAMPLES = ROOT / "samples"

SEED = 42
N_FRAMES = 16
FRAME_SUBSETS = (4, 8, 16)
FRAME_SIZE = 256          # encoder transforms resize to their own input size
BURST_LEN = 8
BURST_STEP_S = 0.25
BURST_SIZE = 64
EDGE_SKIP = 0.05

VIDEOMME_PARQUET_URL = (
    "https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/"
    "videomme/test-00000-of-00001.parquet"
)

CLASSES = [
    "gameplay_animation",
    "sports_broadcast",
    "vlog_egocentric",
    "talking_head",
    "presentation_screen",
    "cinematic_documentary",
]

CHUNKERS = {
    "gameplay_animation": "shot_boundary",
    "sports_broadcast": "motion_segment",
    "vlog_egocentric": "activity",
    "talking_head": "speaker_turn",
    "presentation_screen": "slide_change",
    "cinematic_documentary": "semantic_scene",
}
FALLBACK_CHUNKER = "fixed_window"

# Default (auto) labels; corrected per video in the contact-sheet review.
SUBCATEGORY_TO_CLASS = {
    "Esports": "gameplay_animation",
    "Animation": "gameplay_animation",
    "Basketball": "sports_broadcast",
    "Football": "sports_broadcast",
    "Athletics": "sports_broadcast",
    "Other Sports": "sports_broadcast",
    "Daily Life": "vlog_egocentric",
    "Travel": "vlog_egocentric",
    "Food": "vlog_egocentric",
    "Handicraft": "vlog_egocentric",
    "Exercise": "vlog_egocentric",
    "Fashion": "vlog_egocentric",
    "News Report": "talking_head",
    "Movie & TV Show": "cinematic_documentary",
    "Documentary": "cinematic_documentary",
    "Stage Play": "cinematic_documentary",
    "Variety Show": "cinematic_documentary",
    "Magic Show": "cinematic_documentary",
    "Acrobatics": "cinematic_documentary",
    "Pet & Animal": "cinematic_documentary",
    "Humanity & History": "cinematic_documentary",
    "Literature & Art": "cinematic_documentary",
    "Biology & Medicine": "cinematic_documentary",
    "Finance & Commerce": "cinematic_documentary",
    "Astronomy": "cinematic_documentary",
    "Geography": "cinematic_documentary",
    "Law": "cinematic_documentary",
    "Life Tip": "cinematic_documentary",
    "Technology": "cinematic_documentary",
    "Multilingual": "cinematic_documentary",
}

# U[true][pred]: share of downstream chunking quality kept when routing `true` to `pred`'s chunker.
# An assumption to validate against retrieval recall; see spec section 5.3.
UTILITY = [
    [1.0, 0.7, 0.3, 0.2, 0.2, 0.4],
    [0.7, 1.0, 0.3, 0.2, 0.2, 0.3],
    [0.3, 0.3, 1.0, 0.3, 0.2, 0.5],
    [0.2, 0.2, 0.3, 1.0, 0.6, 0.3],
    [0.2, 0.2, 0.2, 0.6, 1.0, 0.2],
    [0.4, 0.3, 0.5, 0.3, 0.2, 1.0],
]

ENCODER_CANDIDATES = {
    "small": [
        ("MobileCLIP2-S0", "dfndr2b"),
        ("MobileCLIP-S1", "datacompdr"),
        ("TinyCLIP-ViT-39M-16-Text-19M", "yfcc15m"),
    ],
    "large": [("ViT-B-32", "laion2b_s34b_b79k")],
}

ZERO_SHOT_PROMPTS = {
    "gameplay_animation": ["a screenshot of a video game", "esports gameplay with a HUD overlay",
                           "a frame from an animated cartoon"],
    "sports_broadcast": ["a TV broadcast of a sports match", "athletes competing in a stadium",
                         "a basketball or football game on television"],
    "vlog_egocentric": ["a first-person handheld vlog", "a person filming their daily life with a phone",
                        "a travel or cooking vlog"],
    "talking_head": ["a news anchor speaking to the camera", "a person talking in an interview",
                     "a podcast host talking"],
    "presentation_screen": ["a presentation slide with text", "a screen recording of a computer",
                            "a lecture with slides and diagrams"],
    "cinematic_documentary": ["a scene from a movie", "a nature documentary shot", "a cinematic film still"],
}

import numpy as np

import config as C

VIDEOMME_SUBCATEGORIES = {
    "Humanity & History", "Literature & Art", "Biology & Medicine", "Finance & Commerce",
    "Astronomy", "Geography", "Law", "Life Tip", "Technology", "Animation", "Movie & TV Show",
    "Documentary", "News Report", "Esports", "Basketball", "Football", "Athletics",
    "Other Sports", "Stage Play", "Magic Show", "Variety Show", "Acrobatics", "Handicraft",
    "Food", "Fashion", "Daily Life", "Travel", "Pet & Animal", "Exercise", "Multilingual",
}


def test_every_videomme_subcategory_maps_to_a_class():
    assert set(C.SUBCATEGORY_TO_CLASS) == VIDEOMME_SUBCATEGORIES
    assert set(C.SUBCATEGORY_TO_CLASS.values()) <= set(C.CLASSES)


def test_classes_chunkers_prompts_consistent():
    assert len(C.CLASSES) == 6
    assert list(C.CHUNKERS) == C.CLASSES
    assert list(C.ZERO_SHOT_PROMPTS) == C.CLASSES


def test_utility_matrix_shape_and_values():
    U = np.array(C.UTILITY)
    assert U.shape == (6, 6)
    assert np.allclose(U, U.T)
    assert np.allclose(np.diag(U), 1.0)
    assert ((U >= 0) & (U <= 1)).all()

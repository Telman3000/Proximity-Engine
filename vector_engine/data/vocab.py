"""Controlled vocabulary for synthetic scene-object records.

Kept deliberately in one place so the corpus generator is fully parametric and
the variety/realism of synthetic data is auditable.
"""

LABELS = [
    "sofa", "armchair", "coffee table", "dining table", "chair", "stool",
    "bookshelf", "wardrobe", "desk", "bed", "nightstand", "dresser",
    "chandelier", "floor lamp", "table lamp", "pendant light",
    "rug", "curtain", "mirror", "framed artwork", "potted plant", "vase",
    "television", "laptop", "monitor", "keyboard", "telephone",
    "refrigerator", "microwave", "oven", "sink", "kettle",
    "column", "staircase", "door", "window", "exit sign", "reception counter",
    "piano", "drum kit", "podium", "projector screen", "banner",
    "trash bin", "fire extinguisher", "clock", "cushion", "blanket",
]

COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "grey",
    "brown", "beige", "gold", "silver", "navy",
]

MATERIALS = [
    "wood", "metal", "glass", "fabric", "leather", "plastic", "stone", "marble",
]

SIZES = ["small", "medium", "large"]

STATES = ["new", "worn", "on", "off", "open", "closed", "clean", "dusty"]

ROOM_TYPES = [
    "living room", "bedroom", "kitchen", "bathroom", "corridor", "lobby",
    "reception", "ballroom", "conference hall", "office", "lounge", "stage",
]

LOCATIONS = [
    "near the window", "by the entrance", "in the corner", "against the wall",
    "in the centre", "next to the staircase", "beside the column",
    "under the chandelier", "along the far wall", "by the reception",
]

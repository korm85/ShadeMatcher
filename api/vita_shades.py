"""
VITA Classical shade reference values in CIELAB (D65, 2-degree observer).
Values sourced from published colorimetric studies of VITA Classical shade guide.
"""

VITA_SHADES = {
    # Reddish-brownish group
    "A1":  {"L": 74.0, "a":  0.6, "b": 16.0, "group": "A"},
    "A2":  {"L": 71.5, "a":  1.5, "b": 18.5, "group": "A"},
    "A3":  {"L": 68.5, "a":  2.8, "b": 20.5, "group": "A"},
    "A3.5":{"L": 65.0, "a":  3.8, "b": 22.0, "group": "A"},
    "A4":  {"L": 61.0, "a":  4.5, "b": 23.5, "group": "A"},

    # Reddish-yellowish group
    "B1":  {"L": 76.5, "a": -0.5, "b": 14.0, "group": "B"},
    "B2":  {"L": 73.5, "a":  0.5, "b": 17.5, "group": "B"},
    "B3":  {"L": 70.0, "a":  1.8, "b": 20.0, "group": "B"},
    "B4":  {"L": 65.5, "a":  2.5, "b": 22.5, "group": "B"},

    # Greyish group
    "C1":  {"L": 73.5, "a": -1.0, "b": 10.0, "group": "C"},
    "C2":  {"L": 69.0, "a": -0.5, "b": 12.5, "group": "C"},
    "C3":  {"L": 65.0, "a":  0.5, "b": 14.0, "group": "C"},
    "C4":  {"L": 60.0, "a":  1.0, "b": 15.5, "group": "C"},

    # Reddish-grey group
    "D2":  {"L": 72.0, "a":  0.8, "b": 13.0, "group": "D"},
    "D3":  {"L": 67.0, "a":  1.5, "b": 15.5, "group": "D"},
    "D4":  {"L": 62.0, "a":  2.0, "b": 17.0, "group": "D"},
}

GROUP_NAMES = {
    "A": "Reddish-brownish",
    "B": "Reddish-yellowish",
    "C": "Greyish",
    "D": "Reddish-grey",
}

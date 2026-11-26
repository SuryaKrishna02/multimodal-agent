"""The unit_convert skill: one table of factors, and temperatures as a special case."""

# Each unit as a multiple of its family's base unit: meter, kilogram, liter, meter per second.
FACTORS = {
    "length": {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "mi": 1609.344,
        "yd": 0.9144,
        "ft": 0.3048,
        "in": 0.0254,
    },
    "mass": {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125},
    "volume": {"l": 1.0, "ml": 0.001, "gal": 3.785411784, "cup": 0.2365882365},
    "speed": {"mps": 1.0, "kmh": 1 / 3.6, "mph": 0.44704},
}
ALIASES = {
    "meter": "m", "meters": "m", "kilometer": "km", "kilometers": "km", "mile": "mi", "miles": "mi",
    "foot": "ft", "feet": "ft", "inch": "in", "inches": "in", "kilogram": "kg", "kilograms": "kg",
    "gram": "g", "grams": "g", "pound": "lb", "pounds": "lb", "liter": "l", "liters": "l", "litre": "l",
    "gallon": "gal", "gallons": "gal", "km/h": "kmh", "mi/h": "mph", "celsius": "c", "fahrenheit": "f", "kelvin": "k",
}  # fmt: skip
TEMPERATURE = {"c", "f", "k"}


def _unit(name: str) -> str:
    key = name.strip().lower().replace("°", "")
    return ALIASES.get(key, key)


def _to_celsius(value: float, unit: str) -> float:
    return {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[unit]


def _from_celsius(value: float, unit: str) -> float:
    return {"c": value, "f": value * 9 / 5 + 32, "k": value + 273.15}[unit]


def run(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a quantity.

    Args:
        value (float): The number.
        from_unit (str): Its unit, for example "mi" or "miles".
        to_unit (str): The unit wanted.

    Returns:
        str: For example "5 mi = 8.047 km".

    Raises:
        ValueError: Unknown units, or units of different families.
    """
    a, b = _unit(from_unit), _unit(to_unit)
    if a in TEMPERATURE and b in TEMPERATURE:
        out = _from_celsius(_to_celsius(value, a), b)
    else:
        family = next((f for f, units in FACTORS.items() if a in units and b in units), None)
        if family is None:
            raise ValueError(f"cannot convert {from_unit} to {to_unit}")
        out = value * FACTORS[family][a] / FACTORS[family][b]
    return f"{value:g} {a} = {round(out, 3):g} {b}"

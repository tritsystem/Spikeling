"""
animal_acoustic_config.py
==========================
Data-driven acoustic-class + environmental-gate config for the Spikeling
animal-call discrimination prototype.

WHY CONFIG-DRIVEN: this started bat-specific (Resonator neurons wired
directly to bat frequencies in a hand-written .spk file). Generalized into
an "animal detector" architecture: adding a new animal class (bird call,
insect stridulation, rodent ultrasonic vocalization, etc.) should be a
config change HERE, not a rewrite of the .spk builder or the driver.

  - animal_gate_spk_builder.py generates the .spk Resonator/LIF network
    FROM this config (one or more Resonator neurons per AcousticClass band).
  - animal_species_classifier.py is the driver -- it also reads this
    config to know which neuron names to listen for, how to aggregate a
    multi-band class into one genus/species-level detection, and which
    ENV_GATE_RULES + REGION_PRIOR_STUB entries apply to which class.

REAL-DATA DISCIPLINE: every numeric fact below is either (a) a real,
citable bioacoustic/field-guidance value, with its source noted inline in
`citation`, or (b) explicitly marked STUB/PLACEHOLDER and left at a
neutral value. Nothing here is invented to "look plausible".

CURRENTLY POPULATED: UK bats only (see the note above CLASSES for why a
second, non-bat class was deliberately not added in this session).
"""
from dataclasses import dataclass, field


@dataclass
class ResonatorBand:
    """One Resonator neuron to instantiate. `band_label` must be a valid
    .spk identifier suffix -- the generated neuron name is
    f"{class_label}_{band_label}" (or just `class_label` when a class has
    exactly one band, for a cleaner name)."""
    band_label: str
    freq_hz: float
    damping: float


@dataclass
class AcousticClass:
    """One animal-call class -- may be a single species (genus_level=False)
    or an explicitly genus/cluster-level group (genus_level=True, meaning:
    any of its bands firing means "this genus/cluster was present", NOT a
    species-level identification -- see Myotis below for why that
    distinction matters and is enforced, not just documented)."""
    label: str          # .spk-identifier-safe, e.g. "CommonPip"
    display_name: str   # human-readable, for reports
    bands: list[ResonatorBand]
    genus_level: bool
    citation: str


@dataclass
class EnvGateRule:
    """A general 'environmental plausibility' gate -- NOT bat-specific
    machinery. `kind` selects which check the driver's ENV_GATE_HANDLERS
    dispatch table runs (see animal_species_classifier.py); `applies_to`
    lists which AcousticClass.label values this rule can suppress. A
    future non-bat class (e.g. a dawn-chorus bird) would add its own
    EnvGateRule(kind="...") rather than hardcoding gating logic per
    species anywhere in the driver."""
    label: str
    kind: str            # "active_months" | "min_temperature_c" | "max_humidity_pct"
    applies_to: list      # list[str] of AcousticClass.label
    params: dict
    citation: str


# ─────────────────────────────────────────────────────────────────────
# ACOUSTIC CLASSES
# ─────────────────────────────────────────────────────────────────────
# A second, non-bat class was deliberately NOT added here with invented
# numbers. The bar was: only add one if a genuinely well-documented,
# unambiguous real acoustic value can be confidently cited -- none cleared
# that bar in this session, so this stays bat-only for now. The config
# shape itself (AcousticClass/ResonatorBand above) is fully general;
# adding a bird/insect/rodent-ultrasonic class later is one new list
# entry with a real cited value, touching no other file.
CLASSES: list[AcousticClass] = [
    AcousticClass(
        label="CommonPip", display_name="Common Pipistrelle",
        bands=[ResonatorBand("Tone", 45000.0, 0.05)],
        genus_level=False,
        citation="Common Pipistrelle peak call frequency ~45kHz "
                 "(established UK bat bioacoustics)",
    ),
    AcousticClass(
        label="SopranoPip", display_name="Soprano Pipistrelle",
        bands=[ResonatorBand("Tone", 55000.0, 0.05)],
        genus_level=False,
        citation="Soprano Pipistrelle peak call frequency ~55kHz -- "
                 "classic confusable pair with Common Pip (separation is "
                 "mostly a frequency-threshold problem around ~47-48kHz)",
    ),
    AcousticClass(
        label="BigBatCluster",
        display_name="Noctule/Serotine/Leisler's (\"Big Bat\" cluster)",
        bands=[ResonatorBand("Tone", 23500.0, 0.05)],
        genus_level=True,
        citation="Nyctalus/Eptesicus cluster ~20-27kHz, louder/longer "
                 "calls -- acoustically distinct, should be the easiest "
                 "cluster to separate",
    ),
    AcousticClass(
        label="Myotis",
        display_name="Myotis genus (Daubenton's, Brandt's, Whiskered, "
                      "Natterer's, etc.)",
        bands=[
            ResonatorBand("LowBand",  40000.0, 0.15),
            ResonatorBand("MidBand",  60000.0, 0.15),
            ResonatorBand("HighBand", 75000.0, 0.15),
        ],
        genus_level=True,
        citation="Myotis genus: steep FM sweep ~80kHz down to ~30-40kHz, "
                 "short duration (~3-5ms) -- genuinely overlapping "
                 "acoustic parameter space between species; a documented "
                 "open problem in bat bioacoustics, not a modeling "
                 "shortfall. Detectable as a genus/cluster, not reliably "
                 "separable to species from call shape alone.",
    ),
]

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENTAL PLAUSIBILITY GATES -- general mechanism (season/
# temperature/humidity gating of any acoustic class's plausibility, not
# hardcoded "bat torpor" logic). Currently populated with the one real,
# well-documented example this project has: UK bat activity season /
# survey-temperature guidance / rain suppression, applied to every
# currently-configured class since they're all bats.
# ─────────────────────────────────────────────────────────────────────
ENV_GATE_RULES: list[EnvGateRule] = [
    EnvGateRule(
        label="ActiveSeason", kind="active_months",
        applies_to=[c.label for c in CLASSES],
        params={"active_months": {4, 5, 6, 7, 8, 9, 10}},  # April-October
        citation="UK bat active season is broadly April-October "
                 "(hibernation roughly Nov-Mar); Bat Conservation Trust "
                 "survey guidance",
    ),
    EnvGateRule(
        label="MinSurveyTemp", kind="min_temperature_c",
        applies_to=[c.label for c in CLASSES],
        params={"min_temperature_c": 10.0},
        citation="Bat Conservation Trust survey guidance recommends "
                 "surveying above ~10C; most UK bat species reduce or "
                 "suspend activity (torpor) below roughly this range",
    ),
    EnvGateRule(
        label="HeavyRainSuppress", kind="max_humidity_pct",
        applies_to=[c.label for c in CLASSES],
        params={"max_humidity_pct": 90.0},
        citation="Heavy rain degrades echolocation (attenuation/clutter) "
                 "and suppresses insect prey activity -- a general field "
                 "observation, not a precise species-level statistic",
    ),
]

# ─────────────────────────────────────────────────────────────────────
# REGIONAL / GEOLOCATION PRIOR -- STUB.
# ─────────────────────────────────────────────────────────────────────
# No real per-class regional species-presence data (e.g. NBN Atlas, GBIF)
# is available in this environment. Every class defaults to a NEUTRAL
# multiplier (1.0 = no bias), so the architecture is wired end-to-end
# (geolocation IS a real input channel feeding the network, see
# RegionPrior_* neurons in the generated .spk) without fabricating
# plausible-sounding regional statistics. Replace individual entries with
# real, cited values (and record the real source in a comment here) before
# trusting this for anything resembling an actual regional ID prior.
REGION_PRIOR_STUB: dict = {c.label: 1.0 for c in CLASSES}

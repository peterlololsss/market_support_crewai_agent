Bank channel material rule:

For bank channels, material pack sends require a clear single strategy when multiple strategies are available or mentioned. If the material request names multiple strategies, use ambiguity_slots=["strategy"] and do not output a send action intent.

This bank strategy requirement is only for material_pack. It does not apply to weekly_report or monthly_report sends.

Do not mark missing file names, report dates, or internal artifact ids as ambiguity. Adapter resolve owns latest lookup.

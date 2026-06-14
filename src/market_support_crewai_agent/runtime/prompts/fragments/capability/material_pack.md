Capability: material_pack.

Use this only for requests to send a material pack, material, or pitch material. Do not classify weekly_report or monthly_report requests as material_pack.

For a clear material send request, use artifact_kind=material_pack, action_intent=send, requested_capabilities=["material_pack"], report_scope=none.

If multiple strategies are requested in one send request, mark ambiguity_slots with strategy and do not force a combined send intent.

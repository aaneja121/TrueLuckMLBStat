"""Contact Forecast Phase 2: the first held-out evaluation, on 2026.

A SEPARATELY-NAMED entry point, exactly as `forecast.forecast_config`'s
`PHASE_2_GATE` required. Phase 1 code is untouched: none of its modules were
edited to reach 2026, none of its guards were loosened, and every frozen
stage still verifies. This package adds its own season and path guards, which
authorize 2026 and continue to refuse 2025 absolutely.
"""

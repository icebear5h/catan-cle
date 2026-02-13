# Deprecated Files

These files are kept for reference but are no longer actively used.

## Superseded Approaches

- **colonist_to_engine.py** - Early attempt at coordinate mapping. Superseded by the realization that Colonist boards are randomly shuffled, making direct mapping impractical.

- **coordinate_mapper.py** - More sophisticated mapping attempt using resource signatures. Still didn't work reliably due to board randomization.

- **replay_to_training.py** - Original training data generator that tried to map through our engine. Superseded by `generate_training_data.py` which works directly with Colonist coordinates.

- **replay_scraper.py** - WebSocket/Playwright approach to scraping replays. Complex and fragile. Superseded by `replay_api_scraper.py` which uses the direct API endpoint.

- **run_pipeline.py** - Original end-to-end orchestrator. Now we use individual scripts with simpler workflows.

## Why We Abandoned Coordinate Mapping

The key insight: Colonist.io shuffles board layouts randomly each game, so there's no fixed mapping between Colonist node IDs and our engine node IDs.

Our solution: Generate training data directly from Colonist's coordinate system. The agent learns strategic *patterns* (e.g., "place on high-pip ore+wheat") that transfer regardless of coordinate system.

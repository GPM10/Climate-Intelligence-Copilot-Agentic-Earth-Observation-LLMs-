# Climate Intelligence Copilot
## AI-Powered Climate Decision Support System

Climate Intelligence Copilot combines satellite imagery, climate datasets, and language-model reasoning to explain environmental change and recommend evidence-based policy actions.

### What This Platform Delivers

A production-ready workflow that unifies:
- Satellite imagery processing for land-use classification and change detection.
- Optional hyperspectral ingestion for high-dimensional sensors and spectral analytics.
- Climate and biodiversity data aggregation across leading public datasets.
- LLM reasoning layers that explain causal drivers and answer natural-language questions.
- Policy intelligence that prioritizes actionable interventions.

**Illustrative workflow**
```
"Which regions in Ireland have rising land degradation and why?"
-> Satellite Agent: detects deforestation patterns
-> Data Agent: retrieves emissions and biodiversity indicators
-> Reasoning Agent: explains drivers (for example, agricultural expansion)
-> Policy Agent: recommends interventions (for example, reforestation or protected areas)
```

---

## Quick Start

### 1. Clone and Set Up
```bash
cd "Climate Intelligence Copilot (Agentic + Earth Observation + LLMs)"

python -m venv venv
venv\\Scripts\\activate        # Windows
# or: source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Populate the following values:
# OPENAI_API_KEY=...
# GEE_PROJECT_ID=...
# SENTINEL_USERNAME=...
# SENTINEL_PASSWORD=...
# HYPERSPECTRAL_DATA_DIR=... (optional)
# HYPERSPECTRAL_API_TOKEN=... (optional)
```

### 3. Run the Examples
```bash
python example.py
python main.py ask --question "What is the emissions trend in Ireland?"
python main.py analyze-region --region Ireland --start-year 2020 --end-year 2024
jupyter notebook notebooks/analysis.ipynb
```

---

## Project Structure

```
src/
|-- agents/
|   |-- base.py                # Base agent class
|   |-- satellite_agent.py     # Land-use and change detection
|   |-- data_agent.py          # Climate data aggregation
|   |-- reasoning_agent.py     # LLM-powered analysis
|   |-- policy_agent.py        # Policy recommendations
|   `-- __init__.py            # Orchestrator (ClimateCopilot)
|-- geospatial/
|   `-- __init__.py            # Geospatial utilities and Earth Engine helpers
|-- data/
|   `-- __init__.py            # Data-processing utilities
`-- utils/
    `-- __init__.py            # Configuration, logging, vector search helpers

config/
|-- settings.yaml              # Core configuration (LLM, GEE, agents)
`-- gee-key.json               # Google Earth Engine credentials

notebooks/
`-- analysis.ipynb             # Walk-through notebook

main.py                        # CLI entry point
example.py                     # Quick-start script
```

---

## Agent Architecture

### Satellite Agent
- Processes Sentinel-2 multispectral imagery.
- Classifies ten land-use types using a ResNet50 backbone.
- Detects multi-temporal change and produces Grad-CAM overlays.
- Computes NDVI, NDBI, NDMI, and related indices.

```python
from agents.satellite_agent import SatelliteAgent
agent = SatelliteAgent()
result = agent.run({
    "image_path": "sentinel2_image.tif",
    "location": "Ireland",
    "reference_image_path": "historical.tif"
})
```

### Data Agent
- Aggregates emissions (EDGAR, CAMS), climate observations (ECMWF, WorldClim), biodiversity indicators (GBIF, IUCN), and land-cover products (ESA-CCI, Copernicus).
- Supports caching and flexible temporal windows.

```python
from agents.data_agent import DataAgent
agent = DataAgent()
result = agent.run({
    "data_type": "emissions",
    "region": "Ireland",
    "temporal_range": [2020, 2024]
})
```

### Reasoning Agent
- Uses OpenAI or Anthropic models to synthesize agent outputs.
- Explains causal drivers, trends, and uncertainties.
- Returns evidence references and confidence scores.

### Policy Agent
- Generates prioritized, cost-aware recommendations.
- Provides resource estimates, timelines, and co-benefits.

### Orchestrator
- `ClimateCopilot` coordinates the specialist agents and returns a consolidated response.

---

## Technology Stack

| Component | Technologies | Purpose |
|-----------|-------------|---------|
| Satellite Data | Sentinelsat, rasterio, GeoPandas | Sentinel-2 acquisition and preprocessing |
| Geospatial | Google Earth Engine, GDAL | Cloud-based and local spatial analysis |
| Machine Learning | PyTorch, ResNet50, Grad-CAM | Classification and explainability |
| Data Processing | Pandas, NumPy, xarray | Climate and emissions analytics |
| LLM Orchestration | LangChain, CrewAI | Agent coordination and tool use |
| Language Models | OpenAI, Anthropic | Natural-language reasoning |
| Vector Database | FAISS | Semantic retrieval |
| Configuration | Pydantic, YAML | Settings management |

---

## Key Features

1. **Satellite Analysis**: Land-use classification with confidence scores, Grad-CAM explainability, and multi-temporal change detection.
2. **Climate Data Integration**: Multi-source aggregation with sectoral breakdowns, anomaly detection, and trend summaries.
3. **LLM Reasoning**: Natural-language answers grounded in agent outputs with evidence citations and confidence metadata.
4. **Policy Recommendations**: ROI-aware action plans, resource estimates, and implementation timelines.
5. **Advanced Capabilities**: Explainability tooling, time-series analytics, change detection, biodiversity metrics, emissions calculations, semantic search, and caching.
6. **Hyperspectral Mode**: Optional ingestion of high-dimensional cubes with PCA/band-selection composites, mean spectral signatures, and change detection support.

---

## Example Queries

```
"Which regions in Ireland have rising land degradation and why?"
-> Activates Satellite + Data + Reasoning + Policy agents

"What are the greenhouse gas trends in Ireland?"
-> Activates Data + Reasoning agents

"How can Ireland reduce deforestation?"
-> Activates Satellite + Data + Reasoning + Policy agents

"How is biodiversity changing in Irish wetlands?"
-> Activates Data + Reasoning + Policy agents

"What is NDVI?"
-> Activates Reasoning agent
```

---

## Hyperspectral Integration

The satellite agent can now operate in a hyperspectral mode when you provide a cube file or in-memory array.

**How to enable**

1. Set the optional hyperspectral values in `.env` (for example, `HYPERSPECTRAL_DATA_DIR` and provider tokens).
2. Configure the `hyperspectral` block in `config/settings.yaml` to choose the data directory, default band selection, RGB mode (`band_selection` or `pca`), and the sampling fraction for spectral signatures.
3. When invoking the satellite agent (directly or via the copilot), include at least one of `hyperspectral_cube_path`, `hyperspectral_array`, or set `sensor="hyperspectral"`. Optional parameters include `hyperspectral_bands`, `hyperspectral_rgb_mode`, and `reference_hyperspectral_cube_path` for change detection.
4. The agent converts the cube to an RGB composite for the ResNet backbone, computes mean spectral signatures, and returns them alongside the usual classification, Grad-CAM explainability, and change metrics.

**Example (direct use)**
```python
from agents.satellite_agent import SatelliteAgent
from utils import Config

config = Config().to_dict()['agents']['satellite']
config['sentinel'] = Config().to_dict().get('sentinel', {})
config['hyperspectral'] = Config().to_dict().get('hyperspectral', {})
agent = SatelliteAgent(config)
result = agent.run({
    "sensor": "hyperspectral",
    "hyperspectral_cube_path": "./data/hyperspectral/sample_cube.npy",
    "hyperspectral_bands": [5, 25, 60]
})
print(result.data['classification'])
print(result.data['spectral_signature'])
```

When calling `ClimateCopilot`, pass the same keys inside the `context` dictionary so the orchestrator forwards them to the satellite agent.

---
## Configuration Overview

`config/settings.yaml` controls runtime behavior:

```yaml
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7

agents:
  satellite:
    model: "resnet50"
    device: "cuda"
    confidence_threshold: 0.7
  data:
    cache_enabled: true
    timeout: 300
  policy:
    use_llm: true
    rule_based_fallback: true

features:
  explainability:
    enabled: true
    method: "grad-cam"
  time_series_analysis: true
  change_detection: true
```

Adjust dataset paths, cache locations, and service credentials to match your deployment.

---

## Installing Dependencies

```bash
pip install -r requirements.txt
```

For a tailored install you can combine packages manually, for example:
```bash
pip install langchain openai torch torchvision        # LLM and ML stack
pip install rasterio geopandas                        # Geospatial stack
pip install google-cloud-storage sentinelsat          # Satellite ingestion
pip install jupyter                                   # Notebook support
```

---

## Google Earth Engine Setup

1. Create a Google Earth Engine account: https://code.earthengine.google.com/
2. Provision a service account in Google Cloud Console and download the JSON key to `config/gee-key.json`.
3. Update `.env` with:
```
GEE_PROJECT_ID=your-project-id
GEE_SERVICE_ACCOUNT_PATH=./config/gee-key.json
GEE_SERVICE_ACCOUNT_EMAIL=service-account@your-project.iam.gserviceaccount.com
```

---

## Roadmap

**Immediate**
- Configure API credentials in `.env`.
- Run `python example.py` to validate the stack.
- Exercise the CLI via `python main.py ask ...` or `python main.py analyze-region ...`.

**Short Term**
- Enable Google Earth Engine workflows end to end.
- Fine-tune the satellite models on organization-specific datasets.
- Add additional climate or emissions datasets.
- Wrap the CLI with a FastAPI or similar service.

**Long Term**
- Real-time satellite monitoring dashboards.
- Multi-language user experiences.
- Mobile companion application.
- Integrations with external policy and permitting platforms.
- Domain-specific LLM fine-tuning.

---

## Additional References

- Sentinel-2 Mission: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- Google Earth Engine: https://developers.google.com/earth-engine
- LangChain Documentation: https://python.langchain.com/
- PyTorch: https://pytorch.org/

---

## License

MIT License. Refer to `LICENSE` for details.

---

## Contributing

We welcome pull requests. Focus areas include:
- Full Google Earth Engine data pipelines.
- Additional machine-learning architectures.
- Web UI and dashboard experiences.
- Performance and latency optimization.

Open an issue to discuss significant changes before submitting a pull request.

---

## Support

Open an issue in this repository for questions or bug reports. For private discussions, contact the development team directly.

---

Built for climate-intelligence practitioners who need operational insight into land, climate, and policy signals.

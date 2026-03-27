# Climate Intelligence Copilot
## 🌍 AI-Powered Climate Decision Support System

Build intelligent climate analysis with satellite data, LLMs, and agentic reasoning.

### 🎯 What This Does

A **production-ready** Climate Intelligence Copilot that combines:
- 🛰️ **Satellite Imagery** → Land-use classification & change detection
- 📊 **Climate Data** → Emissions, biodiversity, precipitation, temperature
- 🧠 **LLM Reasoning** → Explains causality and answers complex questions
- 🧾 **Policy Agent** → Recommends evidence-based interventions

**Example Query:**
```
"Which regions in Ireland have rising land degradation and why?"
↓
Satellite Agent: Detects deforestation patterns
Data Agent: Retrieves emissions & biodiversity data
Reasoning Agent: Explains drivers (agriculture expansion, logging, etc.)
Policy Agent: Suggests interventions (reforestation, protected areas, etc.)
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd "Climate Intelligence Copilot (Agentic + Earth Observation + LLMs)"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add:
# - OPENAI_API_KEY=sk-...
# - GEE_PROJECT_ID=your-project
# - SENTINEL_USERNAME/PASSWORD
```

### 3. Run Examples
```bash
# Quick demo
python example.py

# CLI interface
python main.py ask --question "What is the emissions trend in Ireland?"
python main.py analyze-region --region Ireland --start-year 2020 --end-year 2024

# Jupyter notebooks
jupyter notebook notebooks/analysis.ipynb
```

---

## 📁 Project Structure

```
src/
├── agents/
│   ├── base.py                 # Base agent class
│   ├── satellite_agent.py      # 🛰️ Land-use & change detection
│   ├── data_agent.py           # 📊 Climate data aggregation
│   ├── reasoning_agent.py      # 🧠 LLM-powered analysis
│   ├── policy_agent.py         # 🧾 Policy recommendations
│   └── __init__.py             # Orchestrator (ClimateCopilot)
├── geospatial/
│   └── __init__.py             # Geospatial utilities, EE integration
├── data/
│   └── __init__.py             # Data processing & calculations
└── utils/
    └── __init__.py             # Config, logging, vector DB

config/
├── settings.yaml               # Configuration (LLM, GEE, agents)
└── gee-key.json               # Google Earth Engine credentials

notebooks/
└── analysis.ipynb             # Jupyter examples

main.py                         # CLI entry point
example.py                      # Quick start example
```

---

## 🧠 Agent Architecture

### **Satellite Agent 🛰️**
- Processes Sentinel-2 multi-spectral imagery
- Classifies 10 land-use types using CNN (ResNet50)
- Detects land-use changes over time
- Provides Grad-CAM visual explanations
- Spectral indices: NDVI, NDBI, NDMI

```python
# Usage
from agents.satellite_agent import SatelliteAgent

agent = SatelliteAgent()
result = agent.run({
    'image_path': 'sentinel2_image.tif',
    'location': 'Ireland',
    'reference_image_path': 'historical.tif'  # For change detection
})

print(result.data['classification'])  # Land use class & confidence
print(result.data['change_detection']) # Change magnitude
```

### **Data Agent 📊**
- Aggregates emissions data (EDGAR, CAMS)
- Fetches climate observations (ECMWF, WorldClim)
- Retrieves biodiversity indicators (GBIF, IUCN)
- Processes land cover data (ESA-CCI, Copernicus)
- Intelligent caching for efficiency

```python
from agents.data_agent import DataAgent

agent = DataAgent()
result = agent.run({
    'data_type': 'emissions',
    'region': 'Ireland',
    'temporal_range': [2020, 2024]
})

print(result.data['emissions'])  # CO2, CH4, N2O time-series
```

### **Reasoning Agent 🧠**
- LLM-powered climate analysis (OpenAI/Anthropic)
- Answers natural language questions
- Explains causality and trends
- Integrates outputs from other agents
- Evidence-based reasoning with confidence scores

```python
from agents.reasoning_agent import ReasoningAgent

agent = ReasoningAgent()
result = agent.run({
    'question': 'Why is deforestation increasing?',
    'context': {'satellite_data': {...}, 'climate_data': {...}}
})

print(result.data['explanation'])
print(result.data['evidence'])
```

### **Policy Agent 🧾**
- Suggests evidence-based interventions
- Prioritizes actions by ROI and feasibility
- Estimates resource requirements and impact
- Creates implementation timelines
- Includes co-benefits analysis

```python
from agents.policy_agent import PolicyAgent

agent = PolicyAgent()
result = agent.run({
    'issue_type': 'deforestation',
    'region': 'Ireland',
    'severity': 'high',
    'context': {...}
})

for rec in result.data['recommendations'][:3]:
    print(f"{rec['policy']} - Priority {rec['priority_rank']}")
```

### **Orchestrator (ClimateCopilot)**
Coordinates all agents for unified intelligence:

```python
from agents import ClimateCopilot
from utils import Config

config = Config()
copilot = ClimateCopilot(config.to_dict())

# Single question triggers appropriate agents
response = copilot.ask(
    question="Which regions have rising land degradation and why?",
    context={'region': 'Ireland', 'temporal_range': [2020, 2024]}
)

# Access all agent outputs
print(response.satellite_analysis.data)
print(response.climate_data.data)
print(response.reasoning.data)
print(response.policy_recommendations.data)
```

---

## 🔧 Tech Stack

| Component | Tech | Purpose |
|-----------|------|---------|
| **Satellite Data** | Sentinelsat, rasterio, geopandas | Sentinel-2 processing |
| **Geospatial** | Google Earth Engine, GDAL | Land-use analysis |
| **ML/Computer Vision** | PyTorch, ResNet50, Grad-CAM | Classification & explanation |
| **Data Processing** | Pandas, NumPy, xarray | Climate data analysis |
| **LLM Orchestration** | LangChain, CrewAI | Agent coordination |
| **LLMs** | OpenAI/Anthropic | Reasoning & explanations |
| **Vector DB** | FAISS | Semantic search over datasets |
| **Config** | Pydantic, YAML | Settings management |

---

## 📊 Key Features

### **1. Satellite Analysis**
```python
# Land-use classification with confidence scores
agent.run({
    'image_path': 'sentinel2.tif',
    'location': 'Ireland'
})

# Output includes:
# - Classification: {class: 'forest', confidence: 0.95}
# - Explainability: Grad-CAM heatmap showing important features
# - Change detection: {change_magnitude: 12.3%, interpretation: 'High change'}
```

### **2. Climate Data Integration**
```python
# Multi-source data aggregation
agent.run({
    'data_type': 'emissions',
    'region': 'Ireland',
    'temporal_range': [2020, 2024]
})

# Output includes:
# - CO2, CH4, N2O time-series with trends
# - Sectoral breakdown
# - Regional comparisons
# - Anomaly detection
```

### **3. LLM-Powered Reasoning**
```python
# Complex question answering with evidence
question = "Why is deforestation increasing in Ireland?"

# Agent automatically:
# - Retrieves satellite data
# - Fetches climate/emissions data
# - Uses LLM to analyze and explain
# - Provides confidence scores
# - Cites evidence sources
```

### **4. Policy Recommendations**
```python
# Intelligent suggestion engine
agent.run({
    'issue_type': 'deforestation',
    'region': 'Ireland',
    'severity': 'high'
})

# Recommendations include:
# - Prioritized by ROI and feasibility
# - Resource estimates
# - Implementation timelines
# - Expected impact & co-benefits
```

### **5. Advanced Features**
- ✅ **Explainability**: Grad-CAM on satellite images
- ✅ **Time-Series Analysis**: Trend detection, anomalies, forecasting
- ✅ **Change Detection**: Multi-temporal satellite comparison
- ✅ **Biodiversity Analysis**: Shannon/Simpson indices, threat assessment
- ✅ **Emissions Calculation**: CO2-equivalent from multiple GHGs
- ✅ **Vector Search**: Semantic similarity search over climate data
- ✅ **Caching**: Intelligent caching for efficiency

---

## 📝 Example Queries

The copilot understands natural language queries:

```python
# Land degradation analysis
"Which regions in Ireland have rising land degradation and why?"
↓ Activates: Satellite + Data + Reasoning + Policy

# Emissions analysis
"What are the greenhouse gas trends in Ireland?"
↓ Activates: Data + Reasoning

# Policy recommendations
"How can Ireland reduce deforestation?"
↓ Activates: Satellite + Data + Reasoning + Policy

# Biodiversity assessment
"How is biodiversity changing in Irish wetlands?"
↓ Activates: Data + Reasoning + Policy

# Simple questions
"What is NDVI?"
↓ Activates: Reasoning
```

---

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
llm:
  provider: "openai"  # openai or anthropic
  model: "gpt-4"
  temperature: 0.7

agents:
  satellite:
    model: "resnet50"
    device: "cuda"  # or cpu
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
    method: "grad-cam"  # grad-cam, saliency, attention
  
  time_series_analysis: true
  change_detection: true
```

---

## 📦 Dependencies Installation

```bash
# Full installation with all features
pip install -r requirements.txt

# Or selective installation:
pip install langchain openai torch torchvision  # For LLM + ML
pip install rasterio geopandas  # For geospatial
pip install google-cloud-storage sentinelsat  # For satellite data
pip install jupyter  # For notebooks
```

---

## 🔐 Setting Up Google Earth Engine

1. **Create GEE Account**: https://code.earthengine.google.com/
2. **Create Service Account**:
   ```
   Google Cloud Console → Service Accounts → Create
   ```
3. **Download JSON Key**: Save as `config/gee-key.json`
4. **Configure in .env**:
   ```
   GEE_PROJECT_ID=your-project-id
   GEE_SERVICE_ACCOUNT_PATH=./config/gee-key.json
   ```

---

## 🚀 Next Steps

### Immediate
- [ ] Configure API keys (.env credentials)
- [ ] Test with `python example.py`
- [ ] Run CLI: `python main.py ask --question "..."`

### Short-term
- [ ] Set up Google Earth Engine integration
- [ ] Fine-tune satellite model on your data
- [ ] Add custom climate datasets
- [ ] Deploy with FastAPI backend

### Long-term
- [ ] Real-time satellite monitoring dashboard
- [ ] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Integration with policy platforms
- [ ] Fine-tuned domain-specific LLM

---

## 📚 Learn More

- **Sentinel-2**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- **Earth Engine**: https://developers.google.com/earth-engine
- **LangChain**: https://python.langchain.com/
- **PyTorch**: https://pytorch.org/

---

## 📄 License

MIT License - See LICENSE file

---

## 🤝 Contributing

Contributions welcome! Areas needing work:
- [ ] Real GEE data integration
- [ ] More advanced ML models
- [ ] Web UI/Dashboard
- [ ] Performance optimization

---

## 📧 Support

For issues and questions, please create an issue or contact the development team.

---

**Built with ❤️ for climate intelligence**

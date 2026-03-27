# Climate Intelligence Copilot - Development & Testing

## Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_agents.py -v
```

## Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## Development Setup

```bash
# Install in development mode with all deps
pip install -e ".[dev,gee,satellite]"

# Jupyter notebooks
jupyter notebook
```

## Structure for New Features

When adding new agents or features:

1. **Agent Implementation**:
   - Inherit from `BaseAgent`
   - Implement: `validate_input()`, `execute()`, `format_output()`
   - Add tests in `tests/test_agents.py`

2. **Utilities**:
   - Add to appropriate module in `src/geospatial/`, `src/data/`, `src/utils/`
   - Include docstrings and type hints
   - Add unit tests

3. **Configuration**:
   - Update `config/settings.yaml`
   - Add env variables to `.env.example`

## Performance Optimization

- Cache results using `src.utils.VectorStoreManager` for semantic search
- Use multiprocessing for batch processing
- Profile with: `python -m cProfile -s cumulative main.py`

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Cold Start Optimization

- Pre-load models on startup
- Cache GEE authentication
- Warm up LLM endpoints

## Troubleshooting

**Issue**: GEE authentication fails
- Solution: Check `config/gee-key.json` path and `GEE_PROJECT_ID`

**Issue**: Out of memory with large satellite images
- Solution: Reduce image resolution or use chunked processing

**Issue**: LLM API timeout
- Solution: Increase timeout in `config/settings.yaml` or use streaming

## Resources

- Architecture: See `docs/architecture.md`
- API Reference: See `docs/api_reference.md`
- Examples: See `notebooks/analysis.ipynb`

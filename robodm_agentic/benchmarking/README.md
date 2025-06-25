# RoboDM Agentic Benchmarking

This directory contains benchmarking tools for the RoboDM Agentic framework, specifically designed to test performance with large-scale robotics datasets.

## DROID Dataset Benchmark

The `droid_benchmark.py` script benchmarks the performance of robodm-agentic with the DROID (Distributed Robotics Open-source Intelligence Dataset) from Google Research.

### Features

- **Parallel Ingestion**: Uses ThreadPoolExecutor to ingest trajectories in parallel
- **Real Model Integration**: Uses actual LLM/VLM models (no mocks)
- **Comprehensive Metrics**: Measures ingestion time, query performance, batch processing
- **Detailed Reporting**: Generates both text reports and JSON metrics
- **Tool Calling**: Uses the new tool calling system instead of code generation

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install tensorflow tensorflow-datasets numpy google-auth
   ```

2. **Install Google Cloud SDK** (for authentication):
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Linux
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Windows
   # Download from https://cloud.google.com/sdk/docs/install
   ```

3. **Set up Google Cloud Authentication**:
   ```bash
   # Run the setup script
   python robodm_agentic/benchmarking/setup_droid_access.py
   
   # Or manually authenticate
   gcloud auth application-default login
   ```

4. **Install Ollama** (for local model inference):
   ```bash
   # Install ollama from https://ollama.ai
   ollama pull qwen2.5:7b
   ollama pull llava:7b
   ```

5. **Alternative: OpenAI** (if you prefer cloud models):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Usage

#### Setup First
```bash
# Run the setup script to configure authentication
python robodm_agentic/benchmarking/setup_droid_access.py
```

#### Test Setup
```bash
# Test the setup without requiring tensorflow
python robodm_agentic/benchmarking/test_benchmark.py
```

#### Basic Usage
```bash
# Run with default settings (1000 trajectories, 4 workers)
python robodm_agentic/benchmarking/droid_benchmark.py
```

#### Custom Configuration
```bash
# Run with custom parameters
python robodm_agentic/benchmarking/droid_benchmark.py \
    --num-trajectories 500 \
    --output-dir ./benchmark_results \
    --max-workers 8
```

#### Start Small
```bash
# Test with a small number of trajectories first
python robodm_agentic/benchmarking/droid_benchmark.py --num-trajectories 10
```

### Command Line Options

- `--num-trajectories`: Number of DROID trajectories to ingest (default: 1000)
- `--output-dir`: Directory to save trajectories and reports (default: temp directory)
- `--max-workers`: Number of parallel workers for ingestion (default: 4)

### Troubleshooting

#### Google Cloud Authentication Issues

If you see errors like:
```
All attempts to get a Google authentication bearer token failed
```

**Solution 1: Use the setup script**
```bash
python robodm_agentic/benchmarking/setup_droid_access.py
```

**Solution 2: Manual authentication**
```bash
# Install Google Cloud SDK
gcloud auth application-default login
```

**Solution 3: Install missing dependencies**
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2
```

#### TensorFlow Import Errors
```bash
pip install tensorflow tensorflow-datasets numpy
```

#### Ollama Connection Issues
```bash
# Ensure ollama is running
ollama serve

# Pull required models
ollama pull qwen2.5:7b
ollama pull llava:7b
```

#### Memory Issues
- Reduce `--num-trajectories` (start with 10-100)
- Reduce `--max-workers` (try 2-4)
- Close other applications to free memory

#### Slow Performance
- Increase `--max-workers` for faster ingestion
- Use SSD storage for better I/O performance
- Ensure good internet connection for dataset download

### Output

The benchmark generates:

1. **Trajectory Files**: Converted DROID trajectories in RoboDM format (`.vla` files)
2. **Benchmark Report**: `benchmark_report.txt` with comprehensive performance analysis
3. **Metrics JSON**: `benchmark_metrics.json` with detailed timing and success data

### Performance Metrics

The benchmark measures:

- **Ingestion Performance**:
  - Total ingestion time
  - Average time per trajectory
  - Parallel processing efficiency
  - Data size statistics

- **Query Performance**:
  - Individual query response times
  - Batch query throughput
  - Success rates
  - Vision analysis performance

- **System Performance**:
  - Memory usage
  - CPU utilization
  - Model inference latency

### Example Output

```
================================================================================
DROID DATASET BENCHMARK REPORT
================================================================================

INGESTION METRICS:
  Total trajectories: 1000
  Total ingestion time: 45.23s
  Average ingestion time per trajectory: 0.045s
  Average trajectory size: 2.34MB
  Total data size: 2340.00MB
  Parallel workers: 4

QUERY PERFORMANCE METRICS:
  Total queries: 15
  Total query time: 67.89s
  Average query time: 4.53s
  Median query time: 3.21s
  Min query time: 1.23s
  Max query time: 12.45s

BATCH PROCESSING METRICS:
  Batch queries: 5
  Batch total time: 18.76s
  Average time per query (batch): 3.75s
  Throughput: 0.27 queries/second

QUERY SUCCESS RATES:
  Successful queries: 14/15
  Success rate: 93.3%
```

### Integration with RoboDM Features

The benchmark leverages RoboDM's optimization features:

- **Compression**: Uses libx264 video codec for efficient storage
- **Parallel Loading**: Tests RoboDM's parallel trajectory loading
- **Frame Selection**: Tests intelligent frame selection for vision queries
- **Metadata Access**: Tests fast metadata retrieval

### Extending the Benchmark

To add new benchmark scenarios:

1. Add new queries to `self.benchmark_queries` in the `DROIDBenchmark` class
2. Implement new metrics in the `metrics` dictionary
3. Add custom analysis in the `generate_report` method
4. Create new benchmark classes for different datasets

### Performance Recommendations

Based on benchmark results, the system provides recommendations for:

- **High Query Times (>5s)**: Implement caching, batch processing, frame selection optimization
- **Moderate Query Times (>2s)**: Consider frame selection optimization and query result caching
- **Good Performance (<2s)**: System is performing well

This benchmark helps identify bottlenecks and optimize the robodm-agentic framework for production use with large-scale robotics datasets. 
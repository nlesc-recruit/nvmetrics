# nvmetrics
## Introduction
nvmetrics is a library designed to facilitate the measurement of GPU metrics using NVIDIA CUPTI (CUDA Profiling Tools Interface). While tools like NCU (NVIDIA Command-Line Profiler) and GUI-based profilers exist, nvmetrics offers a programmatic approach to GPU profiling, enabling seamless integration into custom workflows and applications.

## Features
- Provides a simple C++ interface for measuring GPU metrics.
- Enables programmatic access to NVIDIA CUPTI for GPU profiling.
- Includes a Python interface using pybind11 for easy integration into Python projects.

## Installation
### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- pybind11 (for Python interface)

### Build from source
1. Clone this repository:
```bash
git clone https://git.astron.nl/RD/recruit/nvmetrics.git
```
2. Navigate to the repository directory:
```bash
cd nvmetrics
```
3. Build the library:
```bash
cmake -S. -Bbuild
make -C build
```
4. (Optional) Install the library:
```bash
make install
```

## Usage
### C++
```cpp
#include <vector>
#include <string>
#include "nvmetrics.h"

using namespace nvmetrics;

int main() {
    std::vector<std::string> metrics = {"metric1", "metric2", "metric3"};
    
    // Start measuring metrics
    double start_time = measureMetricsStart(metrics);
    
    // Perform GPU operations
    
    // Stop measuring metrics and retrieve results
    std::vector<double> results = measureMetricsStop();
    
    // Process results
    
    return 0;
}
```

### Python
```python
import nvmetrics

# Define metrics to measure
metrics = ["metric1", "metric2", "metric3"]

# Start measuring metrics
nvmetrics.measureMetricsStart(metrics)

# Perform GPU operations

# Stop measuring metrics and retrieve results
results = nvmetrics.measureMetricsStop()

# Process results
```
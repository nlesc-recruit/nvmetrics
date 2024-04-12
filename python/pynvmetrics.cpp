#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nv_metrics.h>

namespace py = pybind11;

PYBIND11_MODULE(nvmetrics, m) {
  m.doc() = "Pybind11 interface for nv_metrics library";

  m.def("measureMetricsStart", &nvmetrics::measureMetricsStart,
        "Start measuring metrics");
  m.def("measureMetricsStop", &nvmetrics::measureMetricsStop,
        "Stop measuring metrics");
}

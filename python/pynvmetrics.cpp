#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nv_metrics.h>

namespace py = pybind11;

PYBIND11_MODULE(nvmetrics, m) {
  m.doc() = "Pybind11 interface for nv_metrics library";

  py::enum_<NVPW_MetricType>(m, "NVPW_MetricType")
      .value("NVPW_METRIC_TYPE_COUNTER", NVPW_METRIC_TYPE_COUNTER)
      .value("NVPW_METRIC_TYPE_RATIO", NVPW_METRIC_TYPE_RATIO)
      .value("NVPW_METRIC_TYPE_THROUGHPUT", NVPW_METRIC_TYPE_THROUGHPUT)
      .export_values();

  m.def("queryMetrics", &nvmetrics::queryMetrics, "Query available metrics",
        py::arg("metricType"), py::arg("deviceNum") = 0);

  m.def("measureMetricsStart", &nvmetrics::measureMetricsStart,
        "Start measuring metrics", py::arg("metricNames"),
        py::arg("deviceNum") = 0);
        
  m.def("measureMetricsStop", &nvmetrics::measureMetricsStop,
        "Stop measuring metrics");
}

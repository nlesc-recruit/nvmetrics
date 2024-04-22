#include <string>
#include <vector>

#ifndef NVPW_MetricType
typedef enum NVPW_MetricType {
  NVPW_METRIC_TYPE_COUNTER,
  NVPW_METRIC_TYPE_RATIO,
  NVPW_METRIC_TYPE_THROUGHPUT
} NVPW_MetricType;
#endif

namespace nvmetrics {
std::vector<std::string> queryMetrics(NVPW_MetricType metricType,
                                      int deviceNum = 0);
double measureMetricsStart(std::vector<std::string> &metricNames,
                           int deviceNum = 0);
std::vector<double> measureMetricsStop();
} // namespace nvmetrics
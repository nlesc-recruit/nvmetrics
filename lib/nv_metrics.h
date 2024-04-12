#include <string>
#include <vector>

namespace nvmetrics {
double measureMetricsStart(std::vector<std::string> newMetricNames);
std::vector<double> measureMetricsStop();
} // namespace nvmetrics
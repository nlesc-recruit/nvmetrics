#include <string>
#include <vector>

namespace nvmetrics {
double measureMetricsStart(std::vector<std::string> &newMetricNames,
                           int deviceNum = 0);
std::vector<double> measureMetricsStop();
} // namespace nvmetrics
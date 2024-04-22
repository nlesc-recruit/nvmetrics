#ifndef NV_METRIC_EVAL_H_
#define NV_METRIC_EVAL_H_

#include <string>
#include <vector>

namespace NV::Metric::Eval {

/* Function to get aggregate metric values
 * @param[in]  chipName                 Chip name for which to get metric values
 * @param[in]  counterDataImage         Counter data image
 * @param[in]  metricNames              List of metrics to read from counter
 * data image
 * @param[out] metricNameValueMap       Metric name value map
 * @param[in] pCounterAvailabilityImage  Pointer to counter availability image
 * queried on target device
 */
std::vector<double>
GetMetricValues(std::string chipName,
                const std::vector<uint8_t> &counterDataImage,
                const std::vector<std::string> &metricNames,
                const uint8_t *pCounterAvailabilityImage = NULL);

} // namespace NV::Metric::Eval

#endif // NV_METRIC_EVAL_H_
#ifndef NV_METRIC_EVAL_H_
#define NV_METRIC_EVAL_H_

#include <string>
#include <vector>

namespace NV::Metric::Eval {

/* clang-format off
 * \brief Function to get aggregate metric values.
 *
 * This function retrieves aggregate metric values for the specified chip and metrics
 * using the provided counter data image and counter availability image.
 *
 * \param[in]  chipName                   The name of the GPU chip for which to get metric values.
 * \param[in]  counterDataImage           The counter data image containing raw counter values.
 * \param[in]  metricNames                A list of metrics to read from the counter data image.
 * \param[out] metricNameValueMap         A map where the metric values will be stored.
 * \param[in]  pCounterAvailabilityImage  A pointer to the counter availability image queried on
 *                                        the target device.
 * clang-format on */
std::vector<double>
GetMetricValues(std::string chipName,
                const std::vector<uint8_t> &counterDataImage,
                const std::vector<std::string> &metricNames,
                const uint8_t *pCounterAvailabilityImage = NULL);

} // namespace NV::Metric::Eval

#endif // NV_METRIC_EVAL_H_
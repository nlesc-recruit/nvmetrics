#ifndef NV_METRIC_CONFIG_H_
#define NV_METRIC_CONFIG_H_

#include <string>
#include <vector>

#include <nvperf_host.h>

namespace NV::Metric::Config {

/* clang-format off
 * \brief Retrieves a metrics evaluator for the specified chip.
 *
 * This function retrieves a metrics evaluator for the specified chip,
 * utilizing the provided scratch buffer and counter availability image.
 *
 * \param[in]  chipName                  The name of the GPU chip for which to get metric values.
 * \param[in]  scratchBuffer             The scratch buffer to be used by the metrics evaluator.
 * \param[in]  pCounterAvailabilityImage A pointer to the counter availability image queried
 *                                       on the target device.
 *
 * \return     metricEvalutor            A pointer to the metrics evaluator.
 * clang-format on */
NVPW_MetricsEvaluator *
GetMetricsEvaluator(std::string chipName, std::vector<uint8_t> &scratchBuffer,
                    const uint8_t *pCounterAvailabilityImage);

/* clang-format off
 * \brief Retrieves raw metric requests for the specified chip and metric names.
 *
 * This function retrieves raw metric requests for the specified chip and metric names
 * by utilizing NVIDIA's Performance Tools Interface (NVPW) for CUDA Metrics.
 * Raw metric requests are essential for subsequent metric collection and analysis.
 *
 * \param[in]  chipName                   The name of the GPU chip for which to get metric values.
 * \param[in]  metricNames                A list of metrics to read from the counter data image.
 * \param[in]  pCounterAvailabilityImage  A pointer to the counter availability image queried on
 *                                        the target device.
 * \return     rawMetricRequests          A vector with the raw metric requests.
 * clang-format on */
std::vector<NVPA_RawMetricRequest>
GetRawMetricRequests(std::string chipName,
                     const std::vector<std::string> &metricNames,
                     const uint8_t *pCounterAvailabilityImage);

/* clang-format off
 * \brief Generates a configuration image for specified GPU chip and metrics.
 *
 * This function generates a configuration image for the specified GPU chip and metrics
 * using NVIDIA's Performance Tools Interface (NVPW) for CUDA Metrics. The configuration
 * image is crucial for subsequent metric collection and analysis.
 *
 * \param[in]  chipName                   The name of the GPU chip for which to get metric values.
 * \param[in]  metricNames                A list of metrics to read from the counter data image.
 * \param[in]  pCounterAvailabilityImage  A pointer to the counter availability image queried on
 *                                        the target device.
 * \return     configImage                A vector with the configuration image.
 * clang-format on */
std::vector<uint8_t> GetConfigImage(std::string chipName,
                                    const std::vector<std::string> &metricNames,
                                    const uint8_t *pCounterAvailabilityImage);

/* clang-format off
 * \brief Retrieves the counter data prefix image for specified GPU chip and metrics.
 *
 * This function retrieves the counter data prefix image for the specified GPU chip
 * and metrics using NVIDIA's Performance Tools Interface (NVPW) for CUDA Metrics.
 * The counter data prefix image is essential for preparing and interpreting raw counter data.
 *
 * \param[in]  chipName                   The name of the GPU chip for which to get metric values.
 * \param[in]  metricNames                A list of metrics to read from the counter data image.
 * \param[in]  pCounterAvailabilityImage  A pointer to the counter availability image queried on
 *                                        the target device.
 * \return     counterDataImagePrefix     A vector with the counter data prefix image.
 * clang-format on */
std::vector<uint8_t>
GetCounterDataPrefixImage(std::string chipName,
                          const std::vector<std::string> &metricNames,
                          const uint8_t *pCounterAvailabilityImage = NULL);

} // namespace NV::Metric::Config

#endif // NV_METRIC_CONFIG_H_
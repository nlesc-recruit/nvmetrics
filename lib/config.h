#ifndef NV_METRIC_CONFIG_H_
#define NV_METRIC_CONFIG_H_

#include <string>
#include <vector>

#include <nvperf_host.h>

namespace NV::Metric::Config {

/* clang-format off
 * \brief Retrieves raw metric requests for the specified chip and metric names.
 *
 * This function retrieves raw metric requests for the specified chip and metric names
 * by utilizing NVIDIA's Performance Tools Interface (NVPW) for CUDA Metrics.
 * Raw metric requests are essential for subsequent metric collection and analysis.
 *
 * \param[in]  chipName[in]               The name of the GPU chip for which to get metric values.
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
 * \param[out] configImage                An output vector where the configuration image will be stored.
 * \param[in]  pCounterAvailabilityImage  A pointer to the counter availability image queried on
 *                                        the target device.
 * clang-format on */
void GetConfigImage(std::string chipName,
                    const std::vector<std::string> &metricNames,
                    std::vector<uint8_t> &configImage,
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
 * \param[out] counterDataImagePrefix     An output vector where the counter data prefix image will be stored.
 * \param[in]  pCounterAvailabilityImage  A pointer to the counter availability image queried on
 *                                        the target device.
 * clang-format on */
void GetCounterDataPrefixImage(std::string chipName,
                               const std::vector<std::string> &metricNames,
                               std::vector<uint8_t> &counterDataImagePrefix,
                               const uint8_t *pCounterAvailabilityImage = NULL);

} // namespace NV::Metric::Config

#endif // NV_METRIC_CONFIG_H_
#ifndef NV_METRIC_CONFIG_H_
#define NV_METRIC_CONFIG_H_

#include <string>
#include <vector>

#include <nvperf_host.h>

namespace NV::Metric::Config {

void GetRawMetricRequests(std::string chipName,
                          const std::vector<std::string> &metricNames,
                          std::vector<NVPA_RawMetricRequest> &rawMetricRequests,
                          const uint8_t *pCounterAvailabilityImage);

void GetConfigImage(std::string chipName,
                    const std::vector<std::string> &metricNames,
                    std::vector<uint8_t> &configImage,
                    const uint8_t *pCounterAvailabilityImage);

void GetCounterDataPrefixImage(std::string chipName,
                               const std::vector<std::string> &metricNames,
                               std::vector<uint8_t> &counterDataImagePrefix,
                               const uint8_t *pCounterAvailabilityImage = NULL);

} // namespace NV::Metric::Config

#endif // NV_METRIC_CONFIG_H_
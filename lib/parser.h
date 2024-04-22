#ifndef NV_METRIC_PARSER_H_
#define NV_METRIC_PARSER_H_

#include <string>

namespace NV::Metric::Parser {
/* clang-format off
 * \brief Parses a metric name string and extracts relevant information.
 *
 * This function parses the given metric name string and extracts the metric name itself,
 * whether instances should be kept, and whether the metric should be isolated.
 *
 * \param[in]  metricName       The metric name string to be parsed.
 * \param[out] reqName          A pointer to a string where the parsed metric name will be stored.
 * \param[out] isolated         A pointer to a boolean indicating whether the metric should be isolated.
 * \param[out] keepInstances    A pointer to a boolean indicating whether metric instances should be kept.
 *
 * \return Returns true if the parsing is successful, false otherwise.
 * clang-format on */
bool ParseMetricNameString(const std::string &metricName, std::string *reqName,
                           bool *isolated, bool *keepInstances);
} // namespace NV::Metric::Parser

#endif // NV_METRIC_PARSER_H_
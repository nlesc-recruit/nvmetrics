#ifndef NV_METRIC_UTILS_H_
#define NV_METRIC_UTILS_H_

#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
namespace NV::Metric::Utils {

void THROW_RUNTIME_ERROR(const char *apiFuncCall, const char *message,
                         const char *file, int line);

void checkNVPWError(NVPA_Status status, const char *apiFuncCall,
                    const char *file, int line);

void checkCUPTIError(CUptiResult status, const char *apiFuncCall,
                     const char *file, int line);

void checkDriverError(CUresult status, const char *apiFuncCall,
                      const char *file, int line);

void checkRuntimeError(cudaError_t status, const char *apiFuncCall,
                       const char *file, int line);

#define NVPW_API_CALL(apiFuncCall)                                             \
  NV::Metric::Utils::checkNVPWError(apiFuncCall, #apiFuncCall, __FILE__,       \
                                    __LINE__)
#define CUPTI_API_CALL(apiFuncCall)                                            \
  NV::Metric::Utils::checkCUPTIError(apiFuncCall, #apiFuncCall, __FILE__,      \
                                     __LINE__)
#define DRIVER_API_CALL(apiFuncCall)                                           \
  NV::Metric::Utils::checkDriverError(apiFuncCall, #apiFuncCall, __FILE__,     \
                                      __LINE__)
#define RUNTIME_API_CALL(apiFuncCall)                                          \
  NV::Metric::Utils::checkRuntimeError(apiFuncCall, #apiFuncCall, __FILE__,    \
                                       __LINE__)

} // namespace NV::Metric::Utils

#endif // NV_METRIC_UTILS_H_
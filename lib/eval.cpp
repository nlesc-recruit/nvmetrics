#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>

#include "eval.h"
#include "parser.h"
#include "utils.h"

namespace NV::Metric::Eval {

std::vector<double>
GetMetricValues(std::string chipName,
                const std::vector<uint8_t> &counterDataImage,
                const std::vector<std::string> &metricNames,
                const uint8_t *pCounterAvailabilityImage) {
  std::vector<double> metricValues;
  if (!counterDataImage.size()) {
    std::cout << "Counter Data Image is empty!\n";
    return metricValues;
  }

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params
      calculateScratchBufferSizeParam = {
          NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculateScratchBufferSizeParam.pChipName = chipName.c_str();
  calculateScratchBufferSizeParam.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
      &calculateScratchBufferSizeParam);

  std::vector<uint8_t> scratchBuffer(
      calculateScratchBufferSizeParam.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams =
      {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
  metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
  metricEvaluatorInitializeParams.pChipName = chipName.c_str();
  metricEvaluatorInitializeParams.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  metricEvaluatorInitializeParams.pCounterDataImage = counterDataImage.data();
  metricEvaluatorInitializeParams.counterDataImageSize =
      counterDataImage.size();
  NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams);
  NVPW_MetricsEvaluator *metricEvaluator =
      metricEvaluatorInitializeParams.pMetricsEvaluator;

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
      NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE};
  getNumRangesParams.pCounterDataImage = counterDataImage.data();
  NVPW_API_CALL(NVPW_CounterData_GetNumRanges(&getNumRangesParams));

  std::string reqName;
  bool isolated = true;
  bool keepInstances = true;
  for (std::string metricName : metricNames) {
    NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated,
                                              &keepInstances);
    NVPW_MetricEvalRequest metricEvalRequest;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
        convertMetricToEvalRequest = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
    convertMetricToEvalRequest.pMetricName = reqName.c_str();
    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize =
        NVPW_MetricEvalRequest_STRUCT_SIZE;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
        &convertMetricToEvalRequest);

    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges;
         ++rangeIndex) {
      NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams =
          {NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE};
      getRangeDescParams.pCounterDataImage = counterDataImage.data();
      getRangeDescParams.rangeIndex = rangeIndex;
      NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams);
      std::vector<const char *> descriptionPtrs(
          getRangeDescParams.numDescriptions);
      getRangeDescParams.ppDescriptions = descriptionPtrs.data();
      NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams);

      std::string rangeName;
      for (size_t descriptionIndex = 0;
           descriptionIndex < getRangeDescParams.numDescriptions;
           ++descriptionIndex) {
        if (descriptionIndex) {
          rangeName += "/";
        }
        rangeName += descriptionPtrs[descriptionIndex];
      }

      NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = {
          NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
      setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
      setDeviceAttribParams.pCounterDataImage = counterDataImage.data();
      setDeviceAttribParams.counterDataImageSize = counterDataImage.size();
      NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams);

      double metricValue;
      NVPW_MetricsEvaluator_EvaluateToGpuValues_Params
          evaluateToGpuValuesParams = {
              NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
      evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
      evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
      evaluateToGpuValuesParams.numMetricEvalRequests = 1;
      evaluateToGpuValuesParams.metricEvalRequestStructSize =
          NVPW_MetricEvalRequest_STRUCT_SIZE;
      evaluateToGpuValuesParams.metricEvalRequestStrideSize =
          sizeof(NVPW_MetricEvalRequest);
      evaluateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
      evaluateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
      evaluateToGpuValuesParams.rangeIndex = rangeIndex;
      evaluateToGpuValuesParams.isolated = true;
      evaluateToGpuValuesParams.pMetricValues = &metricValue;
      NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateToGpuValuesParams);

      metricValues.push_back(metricValue);
    }
  }

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = {
      NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
  NVPW_API_CALL(NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));

  return metricValues;
}

} // namespace NV::Metric::Eval
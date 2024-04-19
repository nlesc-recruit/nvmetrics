#include "config.h"
#include "eval.h"
#include "utils.h"
#include <cupti_target.h>
#include <functional>
#include <nvperf_host.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace {
static int numRanges = 2;

CUcontext cuContext;

CUdevice cuDevice;
std::string chipName;
std::vector<std::string> metricNames;

std::vector<uint8_t> counterDataImage;
std::vector<uint8_t> counterDataImagePrefix;
std::vector<uint8_t> configImage;
std::vector<uint8_t> counterDataScratchBuffer;
std::vector<uint8_t> counterAvailabilityImage;

bool CreateCounterDataImage(std::vector<uint8_t> &counterDataImage,
                            std::vector<uint8_t> &counterDataScratchBuffer,
                            std::vector<uint8_t> &counterDataImagePrefix) {

  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
  counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
  counterDataImageOptions.maxNumRanges = numRanges;
  counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
  counterDataImageOptions.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_API_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
      CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initializeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  counterDataImage.resize(calculateSizeParams.counterDataImageSize);
  initializeParams.pCounterDataImage = &counterDataImage[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
      scratchBufferSizeParams = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratchBufferSizeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;
  scratchBufferSizeParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
      &scratchBufferSizeParams));

  counterDataScratchBuffer.resize(
      scratchBufferSizeParams.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
      initScratchBufferParams = {
          CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  initScratchBufferParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  initScratchBufferParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize =
      scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer =
      &counterDataScratchBuffer[0];

  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));

  return true;
}

bool runTestStart(CUdevice cuDevice, std::vector<uint8_t> &configImage,
                  std::vector<uint8_t> &counterDataScratchBuffer,
                  std::vector<uint8_t> &counterDataImage,
                  CUpti_ProfilerReplayMode profilerReplayMode,
                  CUpti_ProfilerRange profilerRange) {
  CUpti_Profiler_BeginSession_Params beginSessionParams = {
      CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  CUpti_Profiler_SetConfig_Params setConfigParams = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
      CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};

  beginSessionParams.ctx = NULL;
  beginSessionParams.counterDataImageSize = counterDataImage.size();
  beginSessionParams.pCounterDataImage = &counterDataImage[0];
  beginSessionParams.counterDataScratchBufferSize =
      counterDataScratchBuffer.size();
  beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
  beginSessionParams.range = profilerRange;
  beginSessionParams.replayMode = profilerReplayMode;
  beginSessionParams.maxRangesPerPass = numRanges;
  beginSessionParams.maxLaunchesPerPass = numRanges;

  CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

  setConfigParams.pConfig = &configImage[0];
  setConfigParams.configSize = configImage.size();

  setConfigParams.passIndex = 0;
  CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
  CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
  return true;
}

bool runTestEnd() {
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
      CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
  CUpti_Profiler_EndSession_Params endSessionParams = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

  return true;
}

} // namespace

namespace nvmetrics {
bool static initialized = false;

void measureMetricsStart(std::vector<std::string>& newMetricNames) {
  if (!initialized) {
    initialized = true;
  }
  int deviceNum = 0;
  int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
  DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      cuDevice));
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      cuDevice));
  if (computeCapabilityMajor < 7) {
    throw std::runtime_error(
        "Sample unsupported on Device with compute capability < 7.0\n");
  }

  metricNames = metricNames;
  counterDataImagePrefix = std::vector<uint8_t>();
  configImage = std::vector<uint8_t>();
  counterDataScratchBuffer = std::vector<uint8_t>();
  counterDataImage = std::vector<uint8_t>();

  CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
  /* Get chip name for the cuda device */
  CUpti_Device_GetChipName_Params getChipNameParams = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  getChipNameParams.deviceIndex = deviceNum;
  CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
  chipName = getChipNameParams.pChipName;

  CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  getCounterAvailabilityParams.ctx = cuContext;
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  counterAvailabilityImage.clear();
  counterAvailabilityImage.resize(
      getCounterAvailabilityParams.counterAvailabilityImageSize);
  getCounterAvailabilityParams.pCounterAvailabilityImage =
      counterAvailabilityImage.data();
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initializeHostParams = {
      NVPW_InitializeHost_Params_STRUCT_SIZE};
  NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
  if (metricNames.size()) {
    if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage,
                                            counterAvailabilityImage.data())) {
      throw std::runtime_error("Failed to create configImage");
    }
    if (!NV::Metric::Config::GetCounterDataPrefixImage(
            chipName, metricNames, counterDataImagePrefix)) {
      throw std::runtime_error("Failed to create counterDataImagePrefix");
    }
  } else {
    throw std::runtime_error("No metrics provided to profile");
  }

  if (!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer,
                              counterDataImagePrefix)) {
    throw std::runtime_error("Failed to create counterDataImage");
  }

  CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;
  CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;

  runTestStart(cuDevice, configImage, counterDataScratchBuffer,
               counterDataImage, profilerReplayMode, profilerRange);
}

std::vector<double> measureMetricsStop() {
  runTestEnd();

  auto results = NV::Metric::Eval::GetMetricValues(chipName, counterDataImage,
                                                   metricNames);
  return results;
}

} // namespace nvmetrics
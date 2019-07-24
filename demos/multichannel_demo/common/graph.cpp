// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>

#include "graph.hpp"
#include "threading.hpp"
#include <cldnn/cldnn_config.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace {

void loadImgToIEGraph(const cv::Mat& img, size_t batch, void* ieBuffer) {
    const int channels = img.channels();
    const int height = img.rows;
    const int width = img.cols;

    float* ieData = reinterpret_cast<float*>(ieBuffer);
    int bOffset = static_cast<int>(batch) * channels * width * height;
    for (int c = 0; c < channels; c++) {
        int cOffset = c * width * height;
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++) {
                ieData[bOffset + cOffset + h * width + w] =
                        static_cast<float>(img.at<cv::Vec3b>(h, w)[c]);
            }
        }
    }
}

}  // namespace

void IEGraph::initNetwork(const std::string& deviceName, bool enableThroughput, const std::string& nstreams, const std::string& nthreads) {
    InferenceEngine::CNNNetReader  netReader;

    netReader.ReadNetwork(modelPath);
    netReader.ReadWeights(weightsPath);

    if (!netReader.isParseSuccess()) {
        throw std::logic_error("Failed to parse model!");
    }

    if (deviceName.find("CPU") != std::string::npos) {
        ie.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), "CPU");
        ie.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), nthreads }}, "CPU");
        ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) }}, "CPU");
        ie.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS), nstreams }}, "CPU");
    }
    if (deviceName.find("GPU") != std::string::npos) {
        if (maxRequests > 1)
            ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "GPU_THROUGHPUT_AUTO" }}, "GPU");
        else ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1" }}, "GPU");
        if ((deviceName.find("MULTI") != std::string::npos) &&
            (deviceName.find("CPU") != std::string::npos)) {
                // multi-device execution with the CPU + GPU performs best with GPU trottling hint,
                // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                ie.SetConfig({{ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" }}, "GPU");
        }
    }
    if (!cpuExtensionPath.empty()) {
        auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(cpuExtensionPath);
        ie.AddExtension(extension_ptr, "CPU");
    }
    if (!cldnnConfigPath.empty()) {
        ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, cldnnConfigPath}}, "GPU");
    }
    /** Setting parameter for collecting per layer metrics **/
    if (printPerfReport) {
        ie.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } });
    }

    // Set batch size
    if (batchSize > 1) {
        auto inShapes = netReader.getNetwork().getInputShapes();
        for (auto& pair : inShapes) {
            auto& dims = pair.second;
            if (!dims.empty()) {
                dims[0] = batchSize;
            }
        }
        netReader.getNetwork().reshape(inShapes);
    }

    InferenceEngine::ExecutableNetwork network;
    network = ie.LoadNetwork(netReader.getNetwork(), deviceName);

    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }
    inputDataBlobName = inputInfo.begin()->first;

    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    outputDataBlobNames.reserve(outputInfo.size());
    for (const auto& i : outputInfo) {
        outputDataBlobNames.push_back(i.first);
    }

    for (size_t i = 0; i < maxRequests; ++i) {
        auto req = network.CreateInferRequestPtr();
        availableRequests.push(req);
    }

    availableRequests.front()->StartAsync();
    availableRequests.front()->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

void IEGraph::start(GetterFunc getterFunc, PostprocessingFunc postprocessingFunc) {
    assert(nullptr != getterFunc);
    assert(nullptr != postprocessingFunc);
    assert(nullptr == getter);
    getter = std::move(getterFunc);
    postprocessing = std::move(postprocessingFunc);
    getterThread = std::thread([&]() {
        std::vector<std::shared_ptr<VideoFrame>> vframes;
        std::vector<cv::Mat> imgsToProc(batchSize);
        while (!terminate) {
            vframes.clear();
            size_t b = 0;
            while (b != batchSize) {
                VideoFrame vframe;
                if (getter(vframe)) {
                    vframes.push_back(std::make_shared<VideoFrame>(vframe));
                    ++b;
                } else {
                    if (terminate) {
                        break;
                    }
                }
            }

            InferenceEngine::InferRequest::Ptr req;
            {
                std::cout << "while (!terminate):: std::unique_lock<std::mutex> lock(mtxAvalableRequests);\n"; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                std::unique_lock<std::mutex> lock(mtxAvalableRequests);
                std::cout << "while (!terminate):: std::unique_lock<std::mutex> lock(mtxAvalableRequests); locked\n"; // no in 1
                condVarAvailableRequests.wait(lock, [&]() {
                    return !availableRequests.empty() || terminate;
                });
                std::cout << "while (!terminate):: std::unique_lock<std::mutex> lock(mtxAvalableRequests); waited\n"; // no in 1
                if (terminate) {
                    break;
                }
                req = std::move(availableRequests.front());
                availableRequests.pop();
            }
            std::cout << "end of while (!terminate)::std::unique_lock<std::mutex> lock(mtxAvalableRequests);\n";

            auto inputBlob = req->GetBlob(inputDataBlobName);
            imgsToProc.resize(batchSize);
            for (size_t i = 0; i < batchSize; i++) {
                if (imgsToProc[i].empty()) {
                    auto& dims = inputBlob->getTensorDesc().getDims();
                    assert(4 == dims.size());
                    auto height = static_cast<int>(dims[2]);
                    auto width  = static_cast<int>(dims[3]);
                    imgsToProc[i] = cv::Mat(height, width, CV_8UC3);
                }
            }

            auto preprocess = [&]() {
                auto buff = inputBlob->buffer();
                float* inputPtr = static_cast<float*>(buff);
                auto loopBody = [&](size_t i) {
                    cv::resize(vframes[i]->frame,
                               imgsToProc[i],
                               imgsToProc[i].size());
                    loadImgToIEGraph(imgsToProc[i], i, inputPtr);
                };
#ifdef USE_TBB
                run_in_arena([&](){
                    tbb::parallel_for<size_t>(0, batchSize, loopBody);
                });
#else
                for (size_t i = 0; i < batchSize; i++) {
                    loopBody(i);
                }
#endif
            };

            if (perfTimerInfer.enabled()) {
                {
                    ScopedTimer st(perfTimerPreprocess);
                    preprocess();
                }
                auto startTime = std::chrono::high_resolution_clock::now();
                req->StartAsync();
                std::cout << "if (perfTimerInfer.enabled()):: std::unique_lock<std::mutex> lock(mtxBusyRequests);\n";
                std::unique_lock<std::mutex> lock(mtxBusyRequests);
                busyBatchRequests.push({std::move(vframes), std::move(req), startTime});
            } else {
                preprocess();
                req->StartAsync();
                std::cout << "if (perfTimerInfer.enabled()) else:: std::unique_lock<std::mutex> lock(mtxBusyRequests);\n";
                std::unique_lock<std::mutex> lock(mtxBusyRequests);
                busyBatchRequests.push({std::move(vframes), std::move(req),
                                    std::chrono::high_resolution_clock::time_point()});
            }
            std::cout << "end of if (perfTimerInfer.enabled()):: std::unique_lock<std::mutex> lock(mtxBusyRequests);\n";
            condVarBusyRequests.notify_one();
        }
    });
}

IEGraph::IEGraph(const InitParams& p):
    perfTimerPreprocess(p.collectStats ? PerfTimer::DefaultIterationsCount : 0),
    perfTimerInfer(p.collectStats ? PerfTimer::DefaultIterationsCount : 0),
    confidenceThreshold(0.5f), batchSize(p.batchSize),
    modelPath(p.modelPath), weightsPath(p.weightsPath),
    cpuExtensionPath(p.cpuExtPath), cldnnConfigPath(p.cldnnConfigPath),
    printPerfReport(p.reportPerf), deviceName(p.deviceName),
    maxRequests(p.maxRequests) {
    assert(p.maxRequests > 0);

    initNetwork(p.deviceName, p.enableThroughput, p.nstreams, p.nthreads);
}

InferenceEngine::SizeVector IEGraph::getInputDims() const {
    assert(!availableRequests.empty());
    auto inputBlob = availableRequests.front()->GetBlob(inputDataBlobName);
    return inputBlob->getTensorDesc().getDims();
}

std::vector<std::shared_ptr<VideoFrame> > IEGraph::getBatchData(cv::Size frameSize) {
    std::vector<std::shared_ptr<VideoFrame>> vframes;
    InferenceEngine::InferRequest::Ptr req;
    std::chrono::high_resolution_clock::time_point startTime;
    {
        std::cout << "IEGraph::getBatchData(cv::Size frameSize):: std::unique_lock<std::mutex> lock(mtxBusyRequests);\n";
        std::unique_lock<std::mutex> lock(mtxBusyRequests);
        condVarBusyRequests.wait(lock, [&]() {
            return !busyBatchRequests.empty();
        });
        vframes = std::move(busyBatchRequests.front().vfPtrVec);
        req = std::move(busyBatchRequests.front().req);
        startTime = std::move(busyBatchRequests.front().startTime);
        busyBatchRequests.pop();
    }
    std::cout << "end of IEGraph::getBatchData(cv::Size frameSize):: std::unique_lock<std::mutex> lock(mtxBusyRequests);\n";

    if (nullptr != req && InferenceEngine::OK == req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
        std::cout << "if (nullptr != req && InferenceEngine::OK == req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY))\n"; // no in 2
        auto detections = postprocessing(req, outputDataBlobNames, frameSize);
        for (decltype(detections.size()) i = 0; i < detections.size(); i ++) {
            vframes[i]->detections = std::move(detections[i]);
        }
        if (perfTimerInfer.enabled()) {
            auto endTime = std::chrono::high_resolution_clock::now();
            perfTimerInfer.addValue(endTime - startTime);
        }
    }
    std::cout << "end of if (nullptr != req && InferenceEngine::OK == req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY))\n"; // no in 2

    if (nullptr != req) {
        std::cout << "if (nullptr != req):: std::unique_lock<std::mutex> lock(mtxAvalableRequests);\n";
        std::unique_lock<std::mutex> lock(mtxAvalableRequests);
        availableRequests.push(std::move(req));
        lock.unlock();
        condVarAvailableRequests.notify_one();
    }
    std::cout << "end of if (nullptr != req):: std::unique_lock<std::mutex> lock(mtxAvalableRequests);\n";

    return vframes;
}

unsigned int IEGraph::getBatchSize() const {
    return static_cast<unsigned int>(batchSize);
}

void IEGraph::setDetectionConfidence(float conf) {
    confidenceThreshold = conf;
}

IEGraph::~IEGraph() {
    std::cout << "IEGraph::~IEGraph()\n";
    terminate = true;
    {
        std::unique_lock<std::mutex> lock(mtxAvalableRequests);
        bool ready = false;
        while (!ready) {
            std::unique_lock<std::mutex> lock(mtxBusyRequests);
            if (!busyBatchRequests.empty()) {
                auto& req = busyBatchRequests.front().req;
                if (nullptr != req) {
                    req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                    availableRequests.push(std::move(req));
                }
                busyBatchRequests.pop();
            }
            if (availableRequests.size() == maxRequests) {
                ready = true;
            }
        }
        if (printPerfReport) {
            slog::info << "Performance counts report" << slog::endl << slog::endl;
            printPerformanceCounts(getFullDeviceName(ie, deviceName));
        }
        condVarAvailableRequests.notify_one();
    }
    if (getterThread.joinable()) {
        getterThread.join();
    }
}

IEGraph::Stats IEGraph::getStats() const {
    return Stats{perfTimerPreprocess.getValue(), perfTimerInfer.getValue()};
}

void IEGraph::printPerformanceCounts(std::string fullDeviceName) {
    ::printPerformanceCounts(*availableRequests.front(), std::cout, fullDeviceName, false);
}

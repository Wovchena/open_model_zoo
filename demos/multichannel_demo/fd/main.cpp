// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/**
* \brief The entry point for the Inference Engine multichannel_face_detection demo application
* \file multichannel_face_detection/main.cpp
* \example multichannel_face_detection/main.cpp
*/
#include <iostream>
#include <vector>
#include <utility>

#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <queue>
#include <chrono>
#include <sstream>
#include <memory>
#include <string>
#include <fstream>
#include <regex>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

#include <opencv2/opencv.hpp>

#include <samples/slog.hpp>

#include <samples/args_helper.hpp>

#include "input.hpp"
#include "multichannel_params.hpp"
#include "multichannel_face_detection_params.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"

namespace {

/**
* \brief This function show a help message
*/
void showUsage() {
    std::cout << std::endl;
    std::cout << "multichannel_face_detection [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -m \"<path>\"                  " << face_detection_model_message<< std::endl;
    std::cout << "      -l \"<absolute_path>\"       " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"       " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"                " << target_device_message << std::endl;
    std::cout << "    -nc                          " << num_cameras << std::endl;
    std::cout << "    -bs                          " << batch_size << std::endl;
    std::cout << "    -n_ir                        " << num_infer_requests << std::endl;
    std::cout << "    -n_iqs                       " << input_queue_size << std::endl;
    std::cout << "    -fps_sp                      " << fps_sampling_period << std::endl;
    std::cout << "    -n_sp                        " << num_sampling_periods << std::endl;
    std::cout << "    -pc                          " << performance_counter_message << std::endl;
    std::cout << "    -t                           " << thresh_output_message << std::endl;
    std::cout << "    -no_show                     " << no_show_processed_video << std::endl;
    std::cout << "    -show_stats                  " << show_statistics << std::endl;
    std::cout << "    -duplicate_num               " << duplication_channel_number << std::endl;
    std::cout << "    -real_input_fps              " << real_input_fps << std::endl;
    std::cout << "    -i                           " << input_video << std::endl;
    std::cout << "    -width                       " << image_width << std::endl;
    std::cout << "    -height                      " << image_height << std::endl;
    std::cout << "    -throughput                  " << enable_throughput << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    if (FLAGS_nc == 0 && FLAGS_i.empty()) {
        throw std::logic_error("Please specify at least one video source(web cam or video file)");
    }
    slog::info << "\tDetection model:           " << FLAGS_m << slog::endl;
    slog::info << "\tDetection threshold:       " << FLAGS_t << slog::endl;
    slog::info << "\tUtilizing device:          " << FLAGS_d << slog::endl;
    if (!FLAGS_l.empty()) {
        slog::info << "\tCPU extension library:     " << FLAGS_l << slog::endl;
    }
    if (!FLAGS_c.empty()) {
        slog::info << "\tCLDNN custom kernels map:  " << FLAGS_c << slog::endl;
    }
    slog::info << "\tBatch size:                " << FLAGS_bs << slog::endl;
    slog::info << "\tNumber of infer requests:  " << FLAGS_n_ir << slog::endl;
    slog::info << "\tNumber of input web cams:  "  << FLAGS_nc << slog::endl;

    return true;
}

struct Face {
    cv::Rect2f rect;
    float confidence;
    unsigned char age;
    unsigned char gender;
    Face(cv::Rect2f r, float c, unsigned char a, unsigned char g): rect(r), confidence(c), age(a), gender(g) {}
};

void drawDetections(cv::Mat& img, const std::vector<Face> detections) {
    for (const Face& f : detections) {
        cv::Rect ri(static_cast<int>(f.rect.x*img.cols), static_cast<int>(f.rect.y*img.rows),
                    static_cast<int>(f.rect.width*img.cols), static_cast<int>(f.rect.height*img.rows));
        cv::rectangle(img, ri, cv::Scalar(255, 0, 0), 2);
    }
}

const size_t DISP_WIDTH  = FLAGS_width;
const size_t DISP_HEIGHT = FLAGS_height;
const size_t MAX_INPUTS  = 25;

struct DisplayParams {
    std::string name;
    cv::Size windowSize;
    cv::Size frameSize;
    size_t count;
    cv::Point points[MAX_INPUTS];
};

DisplayParams prepareDisplayParams(size_t count) {
    DisplayParams params;
    params.count = count;
    params.windowSize = cv::Size(DISP_WIDTH, DISP_HEIGHT);
    params.name = FLAGS_title;

    size_t gridCount = static_cast<size_t>(ceil(sqrt(count)));
    size_t gridStepX = static_cast<size_t>(DISP_WIDTH/gridCount);
    size_t gridStepY = static_cast<size_t>(DISP_HEIGHT/gridCount);
    params.frameSize = cv::Size(gridStepX, gridStepY);

    for (size_t i = 0; i < count; i++) {
        cv::Point p;
        p.x = gridStepX * (i/gridCount);
        p.y = gridStepY * (i%gridCount);
        params.points[i] = p;
    }
    return params;
}

void displayNSources(const std::vector<std::shared_ptr<VideoFrame>>& data,
                     float time,
                     const std::string& stats,
                     DisplayParams params) {
    cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
    auto loopBody = [&](size_t i) {
        auto& elem = data[i];
        if (!elem->frame.empty()) {
            cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
            cv::Mat windowPart = windowImage(rectFrame);
            cv::resize(elem->frame, windowPart, params.frameSize);
            drawDetections(windowPart, elem->detections.get<std::vector<Face>>());
        }
    };

    auto drawStats = [&]() {
        if (FLAGS_show_stats && !stats.empty()) {
            static const cv::Point posPoint = cv::Point(3*DISP_WIDTH/4, 4*DISP_HEIGHT/5);
            auto pos = posPoint + cv::Point(0, 25);
            size_t currPos = 0;
            while (true) {
                auto newPos = stats.find('\n', currPos);
                cv::putText(windowImage, stats.substr(currPos, newPos - currPos), pos, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,  cv::Scalar(0, 0, 255), 1);
                if (newPos == std::string::npos) {
                    break;
                }
                pos += cv::Point(0, 25);
                currPos = newPos + 1;
            }
        }
    };

//  #ifdef USE_TBB
#if 0  // disable multithreaded rendering for now
    run_in_arena([&](){
        tbb::parallel_for<size_t>(0, data.size(), [&](size_t i) {
            loopBody(i);
        });
    });
#else
    for (size_t i = 0; i < data.size(); ++i) {
        loopBody(i);
    }
#endif
    drawStats();
    if (0 != time) {
    char str[256];
        double fps = static_cast<double>(1000.0f/time);
        snprintf(str, sizeof(str), "%s: %5.2f fps", FLAGS_title.c_str(), fps);
        cv::putText(windowImage, str, cv::Point(50, 100), cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 1.5,  cv::Scalar(0, 255, 0), 2);
    } else {
       cv::putText(windowImage, FLAGS_title.c_str(), cv::Point(50, 100), cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 255, 0), 2); 
    }
    cv::imshow(params.name, windowImage);
}

std::pair<std::vector<unsigned long>, std::vector<unsigned long>> getCpuStat() {  // returns (idle, total)
    static unsigned ncpu = 0;
    if (0 == ncpu) {
        std::ifstream cpuinfo("/proc/cpuinfo");
        ncpu = std::count(std::istream_iterator<std::string>(cpuinfo),
                          std::istream_iterator<std::string>(),
                          std::string("processor"));
    }
    std::vector<unsigned long> idle(ncpu), total(ncpu);
    std::ifstream proc_stat("/proc/stat");
    while(true)
    {
        std::string line;
        std::getline(proc_stat, line);
        if (!proc_stat.good())
        {
            break;
        }

        std::regex      core_jiffies("^cpu(\\d+)\\s+"   "(\\d+)\\s+" // user
                                                   "(\\d+)\\s+" // nice
                                                   "(\\d+)\\s+" // system
                                                   "(\\d+)\\s+" // idle
                                                   "(\\d+)\\s+" // iowait
                                                   "(\\d+)\\s+" // irq
                                                   "(\\d+)\\s+" // softirq
                                                   "(\\d+)\\s+" // steal
                                                   "(\\d+)\\s+" // guest
                                                   "(\\d+)$");  // guest_nice
        std::smatch     match;

        if ( std::regex_match(line, match, core_jiffies) )
        {

            auto        idleInfo =      stoul(match[5]) + stoul(match[6]),
                        non_idleinfo =  stoul(match[2]) + stoul(match[3]) +
                                    stoul(match[4]) + stoul(match[7]) +
                                    stoul(match[8]) + stoul(match[9]),
                        core_id =   stoul(match[1]);

            idle[core_id] = idleInfo;
            total[core_id] = idleInfo + non_idleinfo;
        }
    }
    return {idle, total};
}

}  // namespace

int main(int argc, char* argv[]) {
	int res = 0;
    try {
#if USE_TBB
        TbbArenaWrapper arena;
#endif
        slog::info << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::string weightsPath;
        std::string modelPath = FLAGS_m;
        std::size_t found = modelPath.find_last_of(".");
        if (found > modelPath.size()) {
            slog::info << "Invalid model name: " << modelPath << slog::endl;
            slog::info << "Expected to be <model_name>.xml" << slog::endl;
            return -1;
        }
        weightsPath = modelPath.substr(0, found) + ".bin";
        slog::info << "Model   path: " << modelPath << slog::endl;
        slog::info << "Weights path: " << weightsPath << slog::endl;

        IEGraph::InitParams graphParams;
        graphParams.batchSize       = FLAGS_bs;
        graphParams.maxRequests     = FLAGS_n_ir;
        graphParams.collectStats    = FLAGS_show_stats;
        graphParams.reportPerf      = FLAGS_pc;
        graphParams.modelPath       = modelPath;
        graphParams.weightsPath     = weightsPath;
        graphParams.cpuExtPath      = FLAGS_l;
        graphParams.cldnnConfigPath = FLAGS_c;
        graphParams.deviceName      = FLAGS_d;
        graphParams.enableThroughput = FLAGS_throughput;
        graphParams.nstreams = FLAGS_nstreams;
        graphParams.nthreads = FLAGS_nthreads;

        std::shared_ptr<IEGraph> network(new IEGraph(graphParams));
        auto inputDims = network->getInputDims();
        if (4 != inputDims.size()) {
            throw std::runtime_error("Invalid network input dimensions");
        }

        std::vector<std::string> files;
        parseInputFilesArguments(files);

        slog::info << "\tNumber of input web cams:    " << FLAGS_nc << slog::endl;
        slog::info << "\tNumber of input video files: " << files.size() << slog::endl;
        slog::info << "\tDuplication multiplayer:     " << FLAGS_duplicate_num << slog::endl;

        const auto duplicateFactor = (1 + FLAGS_duplicate_num);
        size_t numberOfInputs = (FLAGS_nc + files.size()) * duplicateFactor;

        DisplayParams params = prepareDisplayParams(numberOfInputs);

        slog::info << "\tNumber of input channels:    " << numberOfInputs << slog::endl;
        if (numberOfInputs > MAX_INPUTS) {
            throw std::logic_error("Number of inputs exceed maximum value [25]");
        }

        VideoSources::InitParams vsParams;
        vsParams.queueSize            = FLAGS_n_iqs;
        vsParams.collectStats         = FLAGS_show_stats;
        vsParams.realFps              = FLAGS_real_input_fps;
        vsParams.expectedHeight = static_cast<unsigned>(inputDims[2]);
        vsParams.expectedWidth  = static_cast<unsigned>(inputDims[3]);

        VideoSources sources(vsParams);
        if (!files.empty()) {
            slog::info << "Trying to open input video ..." << slog::endl;
            for (auto& file : files) {
                try {
                    sources.openVideo(file, false);
                } catch (...) {
                    slog::info << "Cannot open video [" << file << "]" << slog::endl;
                    throw;
                }
            }
        }
        if (FLAGS_nc) {
            slog::info << "Trying to connect " << FLAGS_nc << " web cams ..." << slog::endl;
            for (size_t i = 0; i < FLAGS_nc; ++i) {
                try {
                    sources.openVideo(std::to_string(i), true);
                } catch (...) {
                    slog::info << "Cannot open web cam [" << i << "]" << slog::endl;
                    throw;
                }
            }
        }
        sources.start();

        size_t currentFrame = 0;

        network->start([&](VideoFrame& img) {
            img.sourceIdx = currentFrame;
            auto camIdx = currentFrame / duplicateFactor;
            currentFrame = (currentFrame + 1) % numberOfInputs;
            return sources.getFrame(camIdx, img);
        }, [](InferenceEngine::InferRequest::Ptr req, const std::vector<std::string>& outputDataBlobNames, cv::Size frameSize) {
            auto output = req->GetBlob(outputDataBlobNames[0]);

            float* dataPtr = output->buffer();
            InferenceEngine::SizeVector svec = output->getTensorDesc().getDims();
            size_t total = 1;
            for (auto v : svec) {
                total *= v;
            }


            std::vector<Detections> detections(getTensorHeight(output->getTensorDesc()) / 200);
            for (auto& d : detections) {
                d.set(new std::vector<Face>);
            }

            for (size_t i = 0; i < total; i+=7) {
                float conf = dataPtr[i + 2];
                if (conf > FLAGS_t) {
                    int idxInBatch = static_cast<int>(dataPtr[i]);
                    float x0 = std::min(std::max(0.0f, dataPtr[i + 3]), 1.0f);
                    float y0 = std::min(std::max(0.0f, dataPtr[i + 4]), 1.0f);
                    float x1 = std::min(std::max(0.0f, dataPtr[i + 5]), 1.0f);
                    float y1 = std::min(std::max(0.0f, dataPtr[i + 6]), 1.0f);

                    cv::Rect2f rect = {x0 , y0, x1-x0, y1-y0};
                    detections[idxInBatch].get<std::vector<Face>>().emplace_back(rect, conf, 0, 0);
                }
            }
            return detections;
        });

        network->setDetectionConfidence(static_cast<float>(FLAGS_t));

        std::atomic<float> averageFps = {0.0f};

        std::vector<std::shared_ptr<VideoFrame>> batchRes;

        std::mutex statMutex;
        std::stringstream statStream;

        const size_t outputQueueSize = 1;
        AsyncOutput output(FLAGS_show_stats, outputQueueSize,
        [&](const std::vector<std::shared_ptr<VideoFrame>>& result) {
            std::string str;
            if (FLAGS_show_stats) {
                std::unique_lock<std::mutex> lock(statMutex);
                str = statStream.str();
            }
            displayNSources(result, averageFps, str, params);

            return cv::waitKey(1);
        });

        output.start();

        using timer = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;
        timer::time_point lastTime = timer::now();
        duration samplingTimeout(FLAGS_fps_sp);

        size_t fpsCounter = 0;

        size_t perfItersCounter = 0;

        cv::namedWindow(params.name, cv::WINDOW_NORMAL);
        cv::resizeWindow(params.name, DISP_WIDTH, DISP_HEIGHT);

        std::pair<std::vector<unsigned long>, std::vector<unsigned long>> prevCpuStat = getCpuStat();

        auto t0 = std::chrono::steady_clock::now();
        unsigned frameCounter = 0;
        unsigned warmUp = 0;

        while (true) {
            bool readData = true;
            while (readData) {
                auto br = network->getBatchData(params.frameSize);
                for (size_t i = 0; i < br.size(); i++) {
                    auto val = static_cast<unsigned int>(br[i]->sourceIdx);
                    auto it = find_if(batchRes.begin(), batchRes.end(), [val] (const std::shared_ptr<VideoFrame>& vf) { return vf->sourceIdx == val; } );
                    if (it != batchRes.end()) {
                        if (!FLAGS_no_show) {
                            output.push(std::move(batchRes));
                        }
                        batchRes.clear();
                        readData = false;
                    }
                    batchRes.push_back(std::move(br[i]));
                }
            }
            if (warmUp > 20) {
                ++frameCounter;
            } else {
                warmUp++;
                if (21 == warmUp) {
                    t0 = std::chrono::steady_clock::now();
                }
            }
            ++fpsCounter;

            if (!output.isAlive()) {
            	res = output.pressedKey;
                break;
            }

            auto currTime = timer::now();
            auto deltaTime = (currTime - lastTime);
            static bool skipFirstReport = true;
            if (deltaTime >= samplingTimeout) {
                auto durMsec =
                        std::chrono::duration_cast<duration>(deltaTime).count();
                auto frameTime = durMsec / static_cast<float>(fpsCounter);
                fpsCounter = 0;
                lastTime = currTime;

                if (FLAGS_no_show) {
                    slog::info << "Average Throughput : " << 1000.f/frameTime << " fps" << slog::endl;
                    if (++perfItersCounter >= FLAGS_n_sp) {
                        break;
                    }
                } else {
                    if (skipFirstReport) {
                        skipFirstReport = false;
                    } else {
                    averageFps = frameTime;
                }
                }

                if (FLAGS_show_stats) {
                    auto inputStat = sources.getStats();
                    auto inferStat = network->getStats();
                    auto outputStat = output.getStats();

                    std::unique_lock<std::mutex> lock(statMutex);
                    statStream.str(std::string());
                    statStream << std::fixed << std::setprecision(1);
                    statStream << "Input reads: ";
                    for (size_t i = 0; i < inputStat.readTimes.size(); ++i) {
                        if (0 == (i % 4)) {
                            statStream << std::endl;
                        }
                        statStream << inputStat.readTimes[i] << "ms ";
                    }
                    statStream << std::endl;
                    statStream << "HW decoding latency: "
                               << inputStat.decodingLatency << "ms";
                    statStream << std::endl;
                    statStream << "Preprocess time: "
                               << inferStat.preprocessTime << "ms";
                    statStream << std::endl;
                    statStream << "Plugin latency: "
                               << inferStat.inferTime << "ms";
                    statStream << std::endl;

                    statStream << "Render time: " << outputStat.renderTime
                               << "ms" << std::endl;

                    if (FLAGS_no_show) {
                        slog::info << statStream.str() << slog::endl;
                    }
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();

        std::pair<std::vector<unsigned long>, std::vector<unsigned long>> currentCpuStat = getCpuStat();

        network.reset();

        std::vector<unsigned long> & prevIdle = prevCpuStat.first, & prevTotal = prevCpuStat.second,
                                   & idle = currentCpuStat.first, & total = currentCpuStat.second;
        std::vector<double> loads(prevIdle.size());
        for (decltype(loads.size()) i = 0; i < prevIdle.size(); i++) {
            double totalInfo = total[i] - prevTotal[i];
            loads[i] = (totalInfo - (idle[i] - prevIdle[i])) / totalInfo * 100;
        }
        double summuryLoad = std::accumulate(loads.begin(), loads.end(), 0.0);
        summuryLoad /= prevIdle.size();
        std::cout << summuryLoad << "% CPU load\n";

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        if (0 != frameCounter) {
            ms meanOverallTimePerAllInputs = std::chrono::duration_cast<ms>(((t1 - t0)) / frameCounter);
            std::cout << "Mean overall time per all inputs: " << std::fixed << std::setprecision(2) << meanOverallTimePerAllInputs.count()
                      << "ms / " << std::chrono::seconds(1) / meanOverallTimePerAllInputs << "FPS\n";
        }
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return res;
}

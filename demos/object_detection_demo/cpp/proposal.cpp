// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>  // TODO: clear headers
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <monitors/presenter.h>
#include <models/detection_model_centernet.h>
#include <models/detection_model_faceboxes.h>
#include <models/detection_model_retinaface.h>
#include <models/detection_model_retinaface_pt.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/args_helper.hpp>
#include <utils/default_flags.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "proposal_common.hpp"
using Processor = ModelBase;  // TODO not now

namespace {
DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char at_message[] = "Required. Architecture type: centernet, faceboxes, retinaface, retinaface-pytorch, ssd or yolo";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
"The demo will look for a suitable plugin for a specified device.";
static const char labels_message[] = "Optional. Path to a file with labels mapping.";
static const char layout_message[] = "Optional. Specify inputs layouts."
" Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";
static const char raw_output_message[] = "Optional. Inference results as raw values.";
static const char input_resizable_message[] = "Optional. Enables resizable input with support of ROI crop & auto resize.";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char iou_thresh_output_message[] = "Optional. Filtering intersection over union threshold for overlapping boxes.";
static const char yolo_af_message[] = "Optional. Use advanced postprocessing/filtering algorithm for YOLO.";
static const char output_resolution_message[] = "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";
static const char anchors_message[] = "Optional. A comma separated list of anchors. "
    "By default used default anchors for model. Only for YOLOV4 architecture type.";
static const char masks_message[] = "Optional. A comma separated list of mask for anchors. "
    "By default used default masks for model. Only for YOLOV4 architecture type.";
static const char reverse_input_channels_message[] = "Optional. Switch the input channels order from BGR to RGB.";
static const char mean_values_message[] = "Optional. Normalize input by subtracting the mean values per channel. Example: \"255.0 255.0 255.0\"";
static const char scale_values_message[] = "Optional. Divide input by scale values per channel. Division is applied "
    "after mean values subtraction. Example: \"255.0 255.0 255.0\"";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(labels, "", labels_message);
DEFINE_string(layout, "", layout_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_double(iou_t, 0.5, iou_thresh_output_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_bool(yolo_af, true, yolo_af_message);
DEFINE_string(output_resolution, "", output_resolution_message);
DEFINE_string(anchors, "", anchors_message);
DEFINE_string(masks, "", masks_message);
DEFINE_bool(reverse_input_channels, false, reverse_input_channels_message);
DEFINE_string(mean_values, "", mean_values_message);
DEFINE_string(scale_values, "", scale_values_message);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    slog::info << ov::get_openvino_version() << slog::endl;
    if (FLAGS_h || 1 == argc) {
    std::cout << "    -h                        " << help_message << std::endl
              << "    -at \"<type>\"              " << at_message << std::endl
              << "    -i                        " << input_message << std::endl
              << "    -m \"<path>\"               " << model_message << std::endl
              << "    -o \"<path>\"               " << output_message << std::endl
              << "    -limit \"<num>\"            " << limit_message << std::endl
              << "    -d \"<device>\"             " << target_device_message << std::endl
              << "    -labels \"<path>\"          " << labels_message << std::endl
              << "    -layout \"<string>\"        " << layout_message << std::endl
              << "    -r                        " << raw_output_message << std::endl
              << "    -t                        " << thresh_output_message << std::endl
              << "    -iou_t                    " << iou_thresh_output_message << std::endl
              << "    -auto_resize              " << input_resizable_message << std::endl
              << "    -nireq \"<integer>\"        " << nireq_message << std::endl
              << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl
              << "    -nstreams                 " << num_streams_message << std::endl
              << "    -loop                     " << loop_message << std::endl
              << "    -no_show                  " << no_show_message << std::endl
              << "    -output_resolution        " << output_resolution_message << std::endl
              << "    -u                        " << utilization_monitors_message << std::endl
              << "    -yolo_af                  " << yolo_af_message << std::endl
              << "    -anchors                  " << anchors_message << std::endl
              << "    -masks                    " << masks_message << std::endl
              << "    -reverse_input_channels   " << reverse_input_channels_message << std::endl
              << "    -mean_values              " << mean_values_message << std::endl
              << "    -scale_values             " << scale_values_message << std::endl;
        showAvailableDevices();
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of -output_resolution parameter is \"width\"x\"height\".");
    }
}

std::unique_ptr<Processor> create_processor(const std::vector<std::string>& labels) {
    std::unique_ptr<ModelBase> processor;
    if (FLAGS_at == "centernet") {
        processor.reset(new ModelCenterNet(FLAGS_m, (float)FLAGS_t, labels, FLAGS_layout));
    } else if (FLAGS_at == "faceboxes") {
        processor.reset(new ModelFaceBoxes(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, (float)FLAGS_iou_t, FLAGS_layout));
    } else if (FLAGS_at == "retinaface") {
        processor.reset(new ModelRetinaFace(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, (float)FLAGS_iou_t, FLAGS_layout));
    } else if (FLAGS_at == "retinaface-pytorch") {
        processor.reset(new ModelRetinaFacePT(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, (float)FLAGS_iou_t, FLAGS_layout));
    } else if (FLAGS_at == "ssd") {
        processor.reset(new ModelSSD(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, labels, FLAGS_layout));
    } else if (FLAGS_at == "yolo") {
        const auto& strAnchors = split(FLAGS_anchors, ',');
        const auto& strMasks = split(FLAGS_masks, ',');

        std::vector<float> anchors;
        std::vector<int64_t> masks;
        try {
            for (auto& str : strAnchors) {
                anchors.push_back(std::stof(str));
            }
        } catch(...) {  // TODO exception
            throw std::runtime_error("Invalid anchors list is provided.");
        }

        try {
            for (auto& str : strMasks) {
                masks.push_back(std::stoll(str));
            }
        }
        catch (...) {  // TODO exception
            throw std::runtime_error("Invalid masks list is provided.");
        }
        processor.reset(new ModelYolo(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, FLAGS_yolo_af, (float)FLAGS_iou_t, labels, anchors, masks, FLAGS_layout));
    } else {
        throw std::runtime_error("No model type or invalid model type (-at) provided: " + FLAGS_at);
    }
    processor->setInputsPreprocessing(FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values);
    return processor;
}

std::vector<ov::InferRequest> create_ireqs(const std::shared_ptr<Processor>& processor) {  // TODO to common
    auto cnnConfig = ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads);
    ov::Core core;
    auto cml = processor->compileModel(cnnConfig, core);
    unsigned int nireq = cnnConfig.maxAsyncRequests;
    if (nireq == 0) {
        try {
            // +1 to use it as a buffer of the pipeline
            nireq = cml.get_property(ov::optimal_number_of_infer_requests) + 1;
        } catch (const ov::Exception& ex) {
            throw std::runtime_error(std::string("Every device used with the demo should support compiled model's property "
                "'OPTIMAL_NUMBER_OF_INFER_REQUESTS'. Failed to query the property with error: ") + ex.what());
        }
    }
    slog::info << "\tNumber of inference requests: " << nireq << slog::endl;
    std::vector<ov::InferRequest> ireqs;
    while (ireqs.size() < nireq) {
        ireqs.push_back(cml.create_infer_request());
    }
    return ireqs;
}

class ColorPalette {
private:
    std::vector<cv::Scalar> palette;

    static double getRandom(double a = 0.0, double b = 1.0) {
        static std::default_random_engine e;
        std::uniform_real_distribution<> dis(a, std::nextafter(b, std::numeric_limits<double>::max()));
        return dis(e);
    }

    static double distance(const cv::Scalar& c1, const cv::Scalar& c2) {
        auto dh = std::fmin(std::fabs(c1[0] - c2[0]), 1 - fabs(c1[0] - c2[0])) * 2;
        auto ds = std::fabs(c1[1] - c2[1]);
        auto dv = std::fabs(c1[2] - c2[2]);

        return dh * dh + ds * ds + dv * dv;
    }

    static cv::Scalar maxMinDistance(const std::vector<cv::Scalar>& colorSet, const std::vector<cv::Scalar>& colorCandidates) {
        std::vector<double> distances;
        distances.reserve(colorCandidates.size());
        for (auto& c1 : colorCandidates) {
            auto min = *std::min_element(colorSet.begin(), colorSet.end(),
                [&c1](const cv::Scalar& a, const cv::Scalar& b) { return distance(c1, a) < distance(c1, b); });
            distances.push_back(distance(c1, min));
        }
        auto max = std::max_element(distances.begin(), distances.end());
        return colorCandidates[std::distance(distances.begin(), max)];
    }

    static cv::Scalar hsv2rgb(const cv::Scalar& hsvColor) {
        cv::Mat rgb;
        cv::Mat hsv(1, 1, CV_8UC3, hsvColor);
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
    }

public:
    explicit ColorPalette(size_t n) {
        palette.reserve(n);
        std::vector<cv::Scalar> hsvColors(1, { 1., 1., 1. });
        std::vector<cv::Scalar> colorCandidates;
        size_t numCandidates = 100;

        hsvColors.reserve(n);
        colorCandidates.resize(numCandidates);
        for (size_t i = 1; i < n; ++i) {
            std::generate(colorCandidates.begin(), colorCandidates.end(),
                []() { return cv::Scalar{ getRandom(), getRandom(0.8, 1.0), getRandom(0.5, 1.0) }; });
            hsvColors.push_back(maxMinDistance(hsvColors, colorCandidates));
        }

        for (auto& hsv : hsvColors) {
            // Convert to OpenCV HSV format
            hsv[0] *= 179;
            hsv[1] *= 255;
            hsv[2] *= 255;

            palette.push_back(hsv2rgb(hsv));
        }
    }

    const cv::Scalar& operator[] (size_t index) const {
        return palette[index % palette.size()];
    }
};

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(DetectionResult& result, const ColorPalette& palette, OutputTransform& outputTransform) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(outputImg);
    // Visualizing result data over source image
    if (FLAGS_r) {
        slog::debug << " -------------------- Frame # " << result.frameId << "--------------------" << slog::endl;
        slog::debug << " Class ID  | Confidence | XMIN | YMIN | XMAX | YMAX " << slog::endl;
    }

    for (auto& obj : result.objects) {
        if (FLAGS_r) {
            slog::debug << " "
                << std::left << std::setw(9) << obj.label << " | "
                << std::setw(10) << obj.confidence << " | "
                << std::setw(4) << int(obj.x) << " | "
                << std::setw(4) << int(obj.y) << " | "
                << std::setw(4) << int(obj.x + obj.width) << " | "
                << std::setw(4) << int(obj.y + obj.height)
                << slog::endl;
        }
        outputTransform.scaleRect(obj);
        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(1) << obj.confidence * 100 << '%';
        const auto& color = palette[obj.labelID];
        putHighlightedText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2);
        cv::rectangle(outputImg, obj, color, 2);
    }

    try {
        for (auto& lmark : result.asRef<RetinaFaceDetectionResult>().landmarks) {
            outputTransform.scaleCoord(lmark);
            cv::circle(outputImg, lmark, 2, cv::Scalar(0, 255, 255), -1);
        }
    }
    catch (const std::bad_cast&) {}

    return outputImg;
}
}  // namespace

int main(int argc, char *argv[]) {
    set_terminate(catcher);
    parse(argc, argv);
    OutputTransform output_transform;
    Presenter presenter(FLAGS_u);
    PerformanceMetrics metrics, preprocMetrics, postprocMetrics, render_metrics;
    std::vector<std::string> labels;
    if (!FLAGS_labels.empty()) {
        labels = DetectionModel::loadLabels(FLAGS_labels);
    }
    ColorPalette palette(labels.size() > 0 ? labels.size() : 100);
    std::shared_ptr<Processor> processor = create_processor(labels);
    OvInferrer<TimedMat> inferrer(create_ireqs(processor));
    std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, FLAGS_nireq == 1 ? read_type::efficient : read_type::safe);
    LazyVideoWriter video_writer{FLAGS_o, cap->fps(), FLAGS_limit};

    for (const auto& state : OvInferrer<TimedMat>::Iterate{inferrer}) {
        TimedMat* timed_mat = state->data();
        if (nullptr == timed_mat) {
            auto start = std::chrono::steady_clock::now();
            const cv::Mat& mat = cap->read();
            if (mat.data) {
                auto preprocStart = std::chrono::steady_clock::now();
                processor->preprocess(ImageInputData{mat}, state->ireq);
                preprocMetrics.update(preprocStart);
                inferrer.submit(std::move(state->ireq), {mat, start});
            } else {
                inferrer.end();
            }
            continue;
        }
        auto post_start = std::chrono::steady_clock::now();
        InferenceResult inf_res;  // TODO conversion operator
        inf_res.metaData = std::make_shared<ImageMetaData>(timed_mat->mat, timed_mat->start);  // TODO align order time and mat args in my meta and theirs
        inf_res.internalModelData = std::make_shared<InternalImageModelData>(timed_mat->mat.cols, timed_mat->mat.rows);
        inf_res.outputsData = {{"conv2d_58/Conv2D/YoloRegion:0", state->ireq.get_tensor("conv2d_58/Conv2D/YoloRegion:0")},  // yolo-v3-tf
        {"conv2d_66/Conv2D/YoloRegion:0", state->ireq.get_tensor("conv2d_66/Conv2D/YoloRegion:0")},
        {"conv2d_74/Conv2D/YoloRegion:0", state->ireq.get_tensor("conv2d_74/Conv2D/YoloRegion:0")}};
        auto postprocStart = std::chrono::steady_clock::now();
        std::unique_ptr<ResultBase> result = processor->postprocess(inf_res);  // TODO process req and timed_mat inhere
        postprocMetrics.update(postprocStart);
        auto render_start = std::chrono::steady_clock::now();
        const cv::Mat& out_im = renderDetectionData(result->asRef<DetectionResult>(), palette, output_transform);
        presenter.drawGraphs(out_im);
        render_metrics.update(render_start);
        metrics.update(timed_mat->start,
            out_im, {10, 22}, cv::FONT_HERSHEY_COMPLEX, 0.65);
        video_writer.write(out_im);
        if (!FLAGS_no_show) {
            cv::imshow(argv[0], out_im);
            int key = cv::pollKey();
            if ('Q' == key || 'q' == key || 27 == key) {  // Esc
                break;
            }
            presenter.handleKey(key);
        }
    }
    slog::info << "Metrics report:" << slog::endl;
    metrics.logTotal();
    logLatencyPerStage(cap->getMetrics().getTotal().latency, preprocMetrics.getTotal().latency,
        inferrer.getInferenceMetircs().getTotal().latency, postprocMetrics.getTotal().latency,
        render_metrics.getTotal().latency);
    slog::info << presenter.reportMeans() << slog::endl;
    return 0;
}

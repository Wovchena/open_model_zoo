// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <models/model_base.h>
#include <models/results.h>
#include <pipelines/metadata.h>  // TODO: remove
#include <utils/config_factory.h>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include <chrono>
#include <forward_list>
#include <random>
#include <queue>
#include <vector>

struct TimedMat {
    cv::Mat mat;
    std::chrono::steady_clock::time_point start;
};

template<typename Meta>
struct OvInferrer {
    struct State {
        State(ov::InferRequest&& ireq): ireq{std::move(ireq)} {}
        virtual ~State() = default;
        virtual Meta* data() {return nullptr;}
        ov::InferRequest ireq;
    };
    struct WithData: State {
        WithData(ov::InferRequest&& ireq, Meta&& meta): State{std::move(ireq)}, meta{std::move(meta)} {}
        Meta meta;
        Meta* data() override {return &meta;}
    };

    struct TimedIreq {
        std::chrono::steady_clock::time_point start;
        ov::InferRequest ireq;
        Meta meta;
    };

    struct Iterate {
        OvInferrer& inferrer;
        struct InputIt {
            OvInferrer& inferrer;

            InputIt(OvInferrer& inferrer) : inferrer{inferrer} {}

            bool operator!=(InputIt) const noexcept {return !inferrer.stop_submit || !inferrer.busy_ireqs.empty();};
            InputIt& operator++() {return *this;}
            std::unique_ptr<State> operator*() {return inferrer.state();}
        };
        InputIt begin() {return inferrer;}
        InputIt end() {return inferrer;}
    };

    std::vector<ov::InferRequest> empty_ireqs;
    std::deque<TimedIreq> busy_ireqs;
    bool stop_submit;
    PerformanceMetrics inferenceMetrics;

    OvInferrer(std::vector<ov::InferRequest> ireqs) : stop_submit{false} {
        for (auto ireq: ireqs) {  // TODO just copy vector
            empty_ireqs.push_back(ireq);
        }
    }

    ~OvInferrer() {
        for (auto& ireq : busy_ireqs) {
            ireq.ireq.cancel();
        }
    }

    void end() {
        if (stop_submit) {
            throw std::runtime_error("Input was over. Unexpected end");
        }
        stop_submit = true;
    }

    void submit(ov::InferRequest&& ireq, Meta&& meta) {
        if (stop_submit) {
            throw std::runtime_error("Input was over. Unexpected submit");
        }
        auto start = std::chrono::steady_clock::now();
        ireq.start_async();
        busy_ireqs.push_back({start, std::move(ireq), std::move(meta)});
    }

    std::unique_ptr<State> state() {
        if (!busy_ireqs.empty() && busy_ireqs.front().ireq.wait_for(std::chrono::milliseconds{0})) {
            TimedIreq timedIreq = std::move(busy_ireqs.front());
            busy_ireqs.pop_front();
            empty_ireqs.push_back(timedIreq.ireq);
            inferenceMetrics.update(timedIreq.start);
            return std::unique_ptr<WithData>(new WithData{std::move(timedIreq.ireq), std::move(timedIreq.meta)});
        }
        if (!(stop_submit || empty_ireqs.empty())) {
            std::unique_ptr<State> res{new State{std::move(empty_ireqs.back())}};
            empty_ireqs.pop_back();
            return res;
        }
        TimedIreq timedIreq = std::move(busy_ireqs.front());
        busy_ireqs.pop_front();
        empty_ireqs.push_back(timedIreq.ireq);
        timedIreq.ireq.wait();
        inferenceMetrics.update(timedIreq.start);
        return std::unique_ptr<WithData>(new WithData{std::move(timedIreq.ireq), std::move(timedIreq.meta)});
    }

    PerformanceMetrics getInferenceMetircs() {return inferenceMetrics;}
};

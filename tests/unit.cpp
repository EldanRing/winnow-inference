#include "planner.h"
#include "protocol.h"
#include <cassert>
#include <iostream>
#include <random>

int main() {
    using namespace winnow;
    // asserts must remain enabled in release test builds.
    auto body = json::parse(
        R"({"state":{"enabled":true},"questions":{"yes":{"type":"noul","instructions":"Is enabled true?"},"route":{"type":"choice","instructions":"Which?","criteria":{"A":null,"B":"second"}},"rating":{"type":"score","instructions":"Level?","criteria":["low","high"]}}})");
    auto c = compile(body, {"A", "B"}, "");
    assert(c.questions.size() == 3);
    assert(c.native["prefix"].get<std::string>().find("State:\n{\"enabled\":true}\n") != std::string::npos);
    assert(c.native["questions"][0]["suffix"].get<std::string>() ==
           "\nQuestion: \"Is enabled true?\"\nOptions:\nA: \"false\"\nB: \"true\"\nReturn the correct letter "
           "label.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Answer:\n");
    auto a = answer(c.questions[0], {0, 0}, 1, false);
    assert(a["noul"] == 0.5 && !a.contains("confidence"));
    auto score = answer(c.questions[2], {0, 0}, 1, false);
    assert(score["score"] == 0.5 && score["confidence"] == 0.0);
    auto plan = make_plan({{1, 2, 3}, {1, 2, 3}, {1, 2, 4}}, {2, 2, 3}, true);
    assert(plan.jobs.size() == 2 && plan.jobs[0].destinations.size() == 2);
    assert(wave_end(plan.jobs, 0, 10, 13, 4) == 1);
    assert(extends({1, 2}, {1, 2, 3}) && !extends({1, 2}, {1, 3}));
    assert(safe_data("<|turn>") == "\"\\u003c|turn>\"");
    TokenCache cache(1024, 2);
    Tokens tokens;
    cache.put("a", {1});
    cache.put("b", {2});
    assert(cache.get("a", tokens));
    cache.put("c", {3});
    assert(!cache.get("b", tokens));
    bool failed = false;
    try {
        compile(json::object(), {"A", "B"}, "");
    } catch (const std::exception &) {
        failed = true;
    }
    assert(failed);
    std::mt19937 random(743);
    for (int trial = 0; trial < 300; ++trial) {
        std::vector<Tokens> inputs;
        std::vector<int> counts;
        Tokens prefix(random() % 70, 1);
        for (unsigned i = 0, n = 1 + random() % 50; i < n; ++i) {
            auto tokens = prefix;
            tokens.push_back(2 + random() % 10);
            inputs.push_back(tokens);
            counts.push_back(2 + random() % 63);
        }
        for (bool optimize : {false, true}) {
            auto plan = make_plan(inputs, counts, optimize);
            std::vector<int> visits(inputs.size());
            for (auto &job : plan.jobs) {
                assert(!job.tokens.empty());
                auto reconstructed = plan.common;
                reconstructed.insert(reconstructed.end(), job.tokens.begin(), job.tokens.end());
                for (auto destination : job.destinations) {
                    assert(reconstructed == inputs[destination]);
                    assert(job.answers >= counts[destination]);
                    ++visits[destination];
                }
            }
            for (auto count : visits)
                assert(count == 1);
            const size_t shared = 100 + plan.common.size(), capacity = shared + 256;
            for (size_t begin = 0; begin < plan.jobs.size();) {
                auto end = wave_end(plan.jobs, begin, shared, capacity, 4);
                assert(end > begin && end - begin <= 4);
                size_t occupied = shared;
                for (auto i = begin; i < end; ++i)
                    occupied += plan.jobs[i].tokens.size();
                assert(occupied <= capacity);
                begin = end;
            }
        }
    }
    std::cout << "Winnow protocol and planner checks passed\n";
}

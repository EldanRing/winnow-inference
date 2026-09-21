#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace winnow {

using Tokens = std::vector<int32_t>;

// A bounded cache of tokenization work, never model answers or cross-state logits.
class TokenCache {
    struct Entry {
        std::string text;
        Tokens tokens;
        size_t bytes;
    };
    size_t used = 0;
    const size_t budget;
    const size_t max_entries;
    std::list<Entry> entries;
    std::unordered_map<std::string, std::list<Entry>::iterator> index;

  public:
    explicit TokenCache(size_t bytes = 8 * 1024 * 1024, size_t count = 256)
        : budget(bytes), max_entries(count) {}
    bool get(const std::string &text, Tokens &result) {
        const auto it = index.find(text);
        if (it == index.end())
            return false;
        entries.splice(entries.begin(), entries, it->second);
        result = it->second->tokens;
        return true;
    }
    void put(const std::string &text, const Tokens &tokens) {
        const size_t bytes = 2 * text.size() + tokens.size() * sizeof(int32_t) + sizeof(Entry);
        if (bytes > budget || max_entries == 0)
            return;
        const auto old = index.find(text);
        if (old != index.end()) {
            used -= old->second->bytes;
            entries.erase(old->second);
            index.erase(old);
        }
        entries.push_front({text, tokens, bytes});
        index.emplace(entries.front().text, entries.begin());
        used += bytes;
        while (used > budget || entries.size() > max_entries) {
            used -= entries.back().bytes;
            index.erase(entries.back().text);
            entries.pop_back();
        }
    }
    size_t bytes() const { return used; }
    void clear() {
        entries.clear();
        index.clear();
        used = 0;
    }
};

struct Job {
    Tokens tokens;
    int answers;
    std::vector<size_t> destinations;
};

struct Plan {
    std::vector<Job> jobs;
    Tokens common;
    size_t original_suffix_tokens = 0;
    size_t unique_suffix_tokens = 0;
};

inline bool extends(const Tokens &previous, const Tokens &next) {
    return !previous.empty() && next.size() >= previous.size() &&
           std::equal(previous.begin(), previous.end(), next.begin());
}

inline Plan make_plan(const std::vector<Tokens> &inputs, const std::vector<int> &counts, bool optimize) {
    if (inputs.empty() || inputs.size() != counts.size())
        throw std::invalid_argument("Invalid question inputs");
    Plan plan;
    std::map<Tokens, size_t> unique;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].empty())
            throw std::invalid_argument("Empty question tokens");
        plan.original_suffix_tokens += inputs[i].size();
        const auto found = unique.find(inputs[i]);
        if (optimize && found != unique.end()) {
            auto &job = plan.jobs[found->second];
            job.destinations.push_back(i);
            job.answers = std::max(job.answers, counts[i]);
        } else {
            unique.emplace(inputs[i], plan.jobs.size());
            plan.jobs.push_back({inputs[i], counts[i], {i}});
            plan.unique_suffix_tokens += inputs[i].size();
        }
    }
    // Retain at least one token per leaf so every unique question produces an output.
    if (optimize && plan.jobs.size() > 1) {
        size_t common = plan.jobs.front().tokens.size() - 1;
        for (const auto &job : plan.jobs) {
            common = std::min(common, job.tokens.size() - 1);
            size_t j = 0;
            while (j < common && plan.jobs.front().tokens[j] == job.tokens[j])
                ++j;
            common = j;
        }
        // Very short prefixes do not justify a separate model dispatch.
        if (common >= 32) {
            plan.common.assign(plan.jobs.front().tokens.begin(), plan.jobs.front().tokens.begin() + common);
            for (auto &job : plan.jobs)
                job.tokens.erase(job.tokens.begin(), job.tokens.begin() + common);
        }
    }
    return plan;
}

inline size_t wave_end(const std::vector<Job> &jobs, size_t begin, size_t shared, size_t capacity,
                       size_t parallel) {
    if (shared >= capacity || parallel == 0)
        throw std::invalid_argument("No question capacity");
    size_t used = shared, end = begin;
    while (end < jobs.size() && end - begin < parallel) {
        if (jobs[end].tokens.size() > capacity - used)
            break;
        used += jobs[end].tokens.size();
        ++end;
    }
    if (end == begin)
        throw std::invalid_argument("State plus question exceeds context capacity");
    return end;
}

} // namespace winnow

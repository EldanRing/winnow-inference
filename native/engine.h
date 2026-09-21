#pragma once
#include "ggml-backend.h"
#include "images.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "nlohmann/json.hpp"
#include "planner.h"
#include <atomic>
#include <cstdlib>
#include <functional>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace winnow {
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Options {
    std::string model, head = "selected", cache = "f16", pipeline = "optimized";
    int context = 32768, parallel = 4, batch = 2048, ubatch = 512, threads = 8;
};

class Engine {
    Options options;
    llama_model *model = nullptr;   // borrowed; the server owns the one weight allocation
    mtmd_context *vision = nullptr; // borrowed projector
    std::function<void()> yield;
    double yield_ms = 0, decode_ms = 0;
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    const llama_vocab *vocab = nullptr;
    ggml_backend_dev_t device = nullptr;
    llama_batch batch{};
    bool batch_ready = false;
    std::string cached_text, device_name;
    std::vector<llama_token> cached_tokens, cached_request_tokens, answer_ids;
    winnow::TokenCache token_cache;
    uint64_t state_id = 0, next_state_id = 1;
    std::vector<std::string> cached_images;
    size_t cached_positions = 0, cached_image_tokens = 0;
    std::vector<float> output_scratch;
    std::vector<std::string> labels;
    std::vector<float> head_rows;
    int width = 0;
    float softcap = 0;

    Images load_images(const json &request) const {
        Images images;
        const auto paths = request.value("images", json::array());
        if (!paths.is_array() || paths.size() > 16)
            throw std::invalid_argument("Expected at most 16 images");
        if (!paths.empty() && !vision)
            throw std::invalid_argument("Images require a matching --mmproj");
        for (const auto &item : paths) {
            auto bytes = image_bytes(item.get<std::string>());
            auto loaded = mtmd_helper_bitmap_init_from_buf(vision, bytes.data(), bytes.size(), false,
                                                           mtmd_helper_init_opt_default());
            Bitmap bitmap(loaded.bitmap, mtmd_bitmap_free);
            if (loaded.video_ctx) {
                mtmd_helper_video_free(loaded.video_ctx);
                throw std::invalid_argument("Only still images are supported");
            }
            if (!bitmap || mtmd_bitmap_is_audio(bitmap.get()))
                throw std::invalid_argument("Could not decode a still image");
            const char *id = mtmd_bitmap_get_id(bitmap.get());
            if (!id || !*id)
                throw std::runtime_error("Missing image content hash");
            images.ids.emplace_back(id);
            images.pointers.push_back(bitmap.get());
            images.owned.push_back(std::move(bitmap));
        }
        return images;
    }
    Chunks image_chunks(const std::string &text, const Images &images) const {
        Chunks chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        // llama-server randomizes its marker; use the borrowed projector's marker.
        std::string marked = text;
        const std::string canonical = "<__media__>";
        const std::string marker = mtmd_get_marker(vision);
        size_t at = 0;
        while ((at = marked.find(canonical, at)) != std::string::npos) {
            marked.replace(at, canonical.size(), marker);
            at += marker.size();
        }
        mtmd_input_text input{marked.data(), marked.size(), true, true};
        if (mtmd_tokenize(vision, chunks.get(), &input, images.pointers.data(), images.pointers.size()))
            throw std::invalid_argument("Image preprocessing failed");
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto chunk = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE &&
                mtmd_decode_use_non_causal(vision, chunk) &&
                mtmd_input_chunk_get_n_tokens(chunk) > llama_n_ubatch(context.get()))
                throw std::invalid_argument(
                    "Image exceeds the microbatch; increase --ubatch-size or resize the image");
        }
        return chunks;
    }
    static size_t image_tokens(const mtmd_input_chunks *chunks) {
        size_t n = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            auto c = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE)
                n += mtmd_input_chunk_get_n_tokens(c);
        }
        return n;
    }
    void prefill_images(const mtmd_input_chunks *chunks) {
        llama_pos position = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const auto chunk = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
                prefill(std::vector<llama_token>(tokens, tokens + n), 0, size_t(position));
                position += n;
            } else {
                const auto t = Clock::now();
                if (mtmd_helper_eval_chunk_single(vision, context.get(), chunk, position, 0, options.batch,
                                                  false, &position))
                    throw std::runtime_error("Image encode/prefill failed");
                llama_synchronize(context.get());
                decode_ms += ms(t);
                cooperate();
            }
        }
        if (position != mtmd_helper_get_n_pos(chunks))
            throw std::runtime_error("Image position mismatch");
        llama_memory_seq_keep(llama_get_memory(context.get()), 0);
    }

    std::vector<llama_token> tokenize(const std::string &text, bool special, bool add) const {
        int size = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, add, special);
        std::vector<llama_token> tokens(size_t(std::abs(size)));
        if (tokens.empty())
            return tokens;
        size = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), add,
                              special);
        if (size < 0)
            throw std::runtime_error("Tokenization failed");
        tokens.resize(size_t(size));
        return tokens;
    }

    void add(llama_token token, int pos, int seq, bool output) {
        const int i = batch.n_tokens++;
        batch.token[i] = token;
        batch.pos[i] = pos;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = output;
    }

    void cooperate() {
        const auto t = Clock::now();
        if (yield)
            yield();
        yield_ms += ms(t);
    }
    void decode() {
        if (!batch.n_tokens)
            return;
        const auto t = Clock::now();
        const int result = llama_decode(context.get(), batch);
        if (result)
            throw std::runtime_error("Model evaluation failed (code " + std::to_string(result) + ")");
        llama_synchronize(context.get());
        decode_ms += ms(t);
        cooperate();
    }

    std::vector<llama_token> cached_tokenize(const std::string &text, size_t &hits) {
        std::vector<llama_token> result;
        if (text.empty())
            return result;
        if (token_cache.get(text, result))
            ++hits;
        else {
            result = tokenize(text, true, false);
            token_cache.put(text, result);
        }
        return result;
    }

    void prefill(const std::vector<llama_token> &tokens, int seq = 0, size_t position = 0, size_t begin = 0) {
        batch.n_tokens = 0;
        for (size_t p = begin; p < tokens.size(); ++p) {
            add(tokens[p], int(position + p), seq, false);
            if (batch.n_tokens == options.batch || p + 1 == tokens.size()) {
                decode();
                batch.n_tokens = 0;
            }
        }
        // llama_decode maintains stream dependencies. Synchronize once at the
        // commit boundary, not once per input chunk plus once per output getter.
        llama_synchronize(context.get());
        if (seq == 0)
            llama_memory_seq_keep(llama_get_memory(context.get()), 0);
    }

    void read_outputs(const std::vector<size_t> &outputs, const winnow::Plan &plan,
                      const std::vector<int> &counts, std::vector<std::vector<float>> &rows) {
        if (outputs.empty())
            return;
        // The bulk getter synchronizes and returns compact rows in batch order.
        const float *values = options.head == "selected" ? llama_get_embeddings(context.get())
                                                         : llama_get_logits(context.get());
        if (!values)
            throw std::runtime_error("Missing model outputs");
        int columns = 0;
        for (auto index : outputs)
            columns = std::max(columns, plan.jobs[index].answers);
        if (options.head == "selected") {
            output_scratch.resize(outputs.size() * size_t(columns));
#ifdef __APPLE__
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(outputs.size()), columns, width, 1.0f,
                        values, width, head_rows.data(), width, 0.0f, output_scratch.data(), columns);
#else
            for (size_t r = 0; r < outputs.size(); ++r)
                for (int c = 0; c < columns; ++c) {
                    double dot = 0;
                    for (int k = 0; k < width; ++k)
                        dot += double(values[r * width + k]) * head_rows[size_t(c) * width + k];
                    output_scratch[r * columns + c] = float(dot);
                }
#endif
        }
        for (size_t r = 0; r < outputs.size(); ++r) {
            const auto &job = plan.jobs[outputs[r]];
            std::vector<float> result(size_t(job.answers));
            for (int i = 0; i < job.answers; ++i) {
                float value = options.head == "selected"
                                  ? output_scratch[r * columns + i]
                                  : values[r * llama_vocab_n_tokens(vocab) + answer_ids[size_t(i)]];
                if (options.head == "selected" && softcap)
                    value = softcap * std::tanh(value / softcap);
                if (!std::isfinite(value))
                    throw std::runtime_error("Non-finite answer logit");
                result[size_t(i)] = value;
            }
            for (const auto destination : job.destinations)
                rows[destination].assign(result.begin(), result.begin() + counts[destination]);
        }
    }

  public:
    Engine(Options opts, llama_model *borrowed, mtmd_context *projector, std::function<void()> cooperate)
        : options(std::move(opts)), model(borrowed), vision(projector), yield(std::move(cooperate)) {
        if (!model)
            throw std::invalid_argument("Missing shared model");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                device = d;
                break;
            }
        }
        if (!device)
            throw std::runtime_error("Winnow requires a GPU backend");
        device_name = ggml_backend_dev_description(device);
        char architecture[128]{};
        llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
        if (std::string(architecture) != "gemma4")
            throw std::invalid_argument("This release supports Gemma 4 GGUF models");
        vocab = llama_model_get_vocab(model);
        std::vector<std::string> candidates;
        for (char a = 'A'; a <= 'Z'; ++a)
            candidates.push_back(std::string(1, a));
        for (char a = 'A'; a <= 'Z'; ++a)
            for (char b = 'A'; b <= 'Z'; ++b)
                candidates.push_back(std::string{a, b});
        for (const auto &label : candidates) {
            const auto ids = tokenize(label, false, false);
            if (ids.size() != 1 ||
                std::find(answer_ids.begin(), answer_ids.end(), ids[0]) != answer_ids.end())
                continue;
            char piece[32];
            const int n = llama_token_to_piece(vocab, ids[0], piece, sizeof(piece), 0, false);
            if (n < 0 || std::string(piece, size_t(n)) != label)
                continue;
            labels.push_back(label);
            answer_ids.push_back(ids[0]);
            if (labels.size() == 64)
                break;
        }
        if (labels.size() < 2)
            throw std::runtime_error("No suitable answer tokens");
        if (options.head == "selected") {
            width = llama_model_n_embd(model);
            head_rows.resize(answer_ids.size() * size_t(width));
            if (llama_model_classifier_rows(model, answer_ids.data(), int(answer_ids.size()),
                                            head_rows.data(), head_rows.size(), &softcap) != width)
                throw std::runtime_error("Unsupported selected output head; use --head full");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = options.context;
        cp.n_batch = options.batch;
        cp.n_ubatch = options.ubatch;
        cp.n_seq_max = options.parallel + 2; // State 0, request prefix 1, question leaves 2..N+1.
        // Runtime initialization reserves one potential output per sequence,
        // including the two non-output prefix sequences.
        cp.n_outputs_max = cp.n_seq_max;
        cp.n_outputs_max_per_seq = 1;
        cp.n_threads = cp.n_threads_batch = options.threads;
        cp.kv_unified = true;
        cp.swa_full = false;
        cp.classifier_only = options.head == "selected";
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = options.cache == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        cp.no_perf = false;
        context.reset(llama_init_from_model(model, cp));
        if (!context)
            throw std::bad_alloc();
#ifdef __APPLE__
        // Metal allows allocations beyond its resident working-set budget.
        // Treat an exhausted GPU budget as an admission failure so the service
        // can release idle chat KV and retry without reducing model context.
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        if (total && !free)
            throw std::bad_alloc();
#endif
        batch = llama_batch_init(options.batch, 0, 1);
        batch_ready = true;
    }

    ~Engine() {
        if (batch_ready)
            llama_batch_free(batch);
    }

    json info() const {
        const int size = llama_model_meta_val_str(model, "tokenizer.chat_template", nullptr, 0);
        std::vector<char> text(size_t(std::max(0, size)) + 1, '\0');
        if (size > 0)
            llama_model_meta_val_str(model, "tokenizer.chat_template", text.data(), text.size());
        return {{"type", "ready"},
                {"protocol", 1},
                {"runtime_sha256", WINNOW_RUNTIME_SHA256},
                {"device", device_name},
                {"architecture", "gemma4"},
                {"chat_template", std::string(text.data())},
                {"labels", labels},
                {"label_token_ids", answer_ids},
                {"context", llama_n_ctx(context.get())},
                {"parallel", options.parallel},
                {"head", options.head},
                {"cache_type", options.cache},
                {"bounded_cache", true},
                {"shared_cache", true},
                {"vision", vision != nullptr},
                {"batch", options.batch},
                {"ubatch", options.ubatch},
                {"pipeline", options.pipeline},
                {"request_prefix_cache", true},
                {"append_state", true}};
    }

    void reset(bool clear_token_cache = true) {
        llama_memory_clear(llama_get_memory(context.get()), true);
        cached_text.clear();
        cached_images.clear();
        cached_positions = cached_image_tokens = 0;
        cached_tokens.clear();
        cached_request_tokens.clear();
        state_id = 0;
        if (clear_token_cache)
            token_cache.clear();
        batch.n_tokens = 0;
    }

    json inspect(const json &request) const {
        const auto text = request.at("prefix").get<std::string>();
        const auto prefix = tokenize(text, true, true);
        auto images = load_images(request);
        Chunks chunks(nullptr, mtmd_input_chunks_free);
        if (!images.ids.empty())
            chunks = image_chunks(text, images);
        const auto shared = tokenize(request.value("request_prefix", std::string()), true, false);
        json result = {
            {"prefix_tokens", chunks ? size_t(mtmd_helper_get_n_pos(chunks.get())) : prefix.size()},
            {"image_tokens", chunks ? image_tokens(chunks.get()) : 0},
            {"request_prefix_tokens", shared.size()},
            {"suffix_tokens", json::array()}};
        const bool ids = request.value("include_token_ids", false);
        if (ids) {
            result["prefix_token_ids"] = prefix;
            result["request_prefix_token_ids"] = shared;
            result["suffix_token_ids"] = json::array();
        }
        for (const auto &q : request.value("questions", json::array())) {
            const auto suffix = tokenize(q.at("suffix").get<std::string>(), true, false);
            result["suffix_tokens"].push_back(suffix.size());
            if (ids)
                result["suffix_token_ids"].push_back(suffix);
        }
        return result;
    }

    json evaluate(const json &request, bool prepare = false) {
        const auto start = Clock::now();
        yield_ms = decode_ms = 0;
        const bool optimize = options.pipeline == "optimized";
        const bool force_cold = !request.value("reuse_prefix", true);
        const std::string *text = nullptr;
        if (request.contains("prefix_id")) {
            if (!state_id || request.at("prefix_id").get<uint64_t>() != state_id)
                throw std::invalid_argument("Unknown native state handle");
            text = &cached_text;
        } else
            text = &request.at("prefix").get_ref<const std::string &>();
        auto images = load_images(request);
        const bool hit =
            !force_cold && !cached_tokens.empty() && cached_text == *text && cached_images == images.ids;
        Chunks chunks(nullptr, mtmd_input_chunks_free);
        if (!hit && !images.ids.empty())
            chunks = image_chunks(*text, images);
        std::vector<llama_token> new_prefix;
        if (!hit)
            new_prefix = tokenize(*text, true, true);
        const size_t prefix_size =
            hit ? cached_positions
                : (chunks ? size_t(mtmd_helper_get_n_pos(chunks.get())) : new_prefix.size());
        if (!prefix_size || prefix_size >= llama_n_ctx(context.get()))
            throw std::invalid_argument("State must leave space for question suffixes inside the context");
        const json empty_questions = json::array();
        const auto &questions = request.contains("questions") ? request.at("questions") : empty_questions;
        if (!questions.is_array() || questions.size() > 256 || (!prepare && questions.empty()))
            throw std::invalid_argument("Expected 1–256 questions");
        const int parallel = request.value("parallel", options.parallel);
        if (parallel < 1 || parallel > options.parallel)
            throw std::invalid_argument("Invalid parallelism");

        size_t token_hits = 0;
        const auto request_prefix =
            cached_tokenize(request.value("request_prefix", std::string()), token_hits);
        std::vector<winnow::Tokens> suffixes;
        std::vector<int> counts;
        for (const auto &q : questions) {
            const int count = q.at("answer_count").get<int>();
            if (count < 2 || count > int(labels.size()))
                throw std::invalid_argument("Invalid answer count");
            auto tokens = cached_tokenize(q.at("suffix").get<std::string>(), token_hits);
            if (tokens.empty())
                throw std::invalid_argument("Empty suffix");
            if (!optimize)
                tokens.insert(tokens.begin(), request_prefix.begin(), request_prefix.end());
            suffixes.push_back(std::move(tokens));
            counts.push_back(count);
        }
        winnow::Plan plan;
        if (!questions.empty())
            plan = winnow::make_plan(suffixes, counts, optimize);
        winnow::Tokens common;
        if (optimize) {
            common = request_prefix;
            common.insert(common.end(), plan.common.begin(), plan.common.end());
        }
        const size_t shared_size = prefix_size + common.size();
        for (size_t wave = 0; wave < plan.jobs.size();)
            wave =
                winnow::wave_end(plan.jobs, wave, shared_size, llama_n_ctx(context.get()), size_t(parallel));

        const double tokenize_ms = ms(start);
        auto memory = llama_get_memory(context.get());
        double prefill_ms = 0;
        size_t reused_prefix_tokens = hit ? prefix_size : 0;
        if (!hit) {
            // Never rewind a bounded local cache: it no longer contains the
            // earlier local window. Reuse only a verified complete token prefix.
            const bool append = optimize && !force_cold && images.ids.empty() && cached_images.empty() &&
                                winnow::extends(cached_tokens, new_prefix);
            const std::string new_text = *text;
            const size_t begin = append ? cached_tokens.size() : 0;
            if (append) {
                llama_memory_seq_keep(memory, 0);
                cached_request_tokens.clear();
                reused_prefix_tokens = begin;
            } else
                reset(false);
            const auto t = Clock::now();
            if (chunks)
                prefill_images(chunks.get());
            else
                prefill(new_prefix, 0, 0, begin);
            prefill_ms = ms(t);
            cached_text = new_text;
            cached_tokens = std::move(new_prefix);
            cached_images = images.ids;
            cached_positions = prefix_size;
            cached_image_tokens = chunks ? image_tokens(chunks.get()) : 0;
            state_id = next_state_id++;
        }

        size_t request_reused_tokens = 0;
        double request_prefill_ms = 0;
        const int parent = common.empty() ? 0 : 1;
        if (common.empty()) {
            if (!cached_request_tokens.empty())
                llama_memory_seq_rm(memory, 1, -1, -1);
            cached_request_tokens.clear();
        } else {
            const bool extension = winnow::extends(cached_request_tokens, common);
            if (extension)
                request_reused_tokens = cached_request_tokens.size();
            else {
                llama_memory_seq_rm(memory, 1, -1, -1);
                llama_memory_seq_cp(memory, 0, 1, -1, -1);
            }
            const auto t = Clock::now();
            if (request_reused_tokens < common.size())
                prefill(common, 1, prefix_size, request_reused_tokens);
            request_prefill_ms = ms(t);
            cached_request_tokens = common;
        }

        std::vector<std::vector<float>> rows(questions.size());
        const auto t = Clock::now();
        size_t waves = 0, leaf_tokens = 0;
        for (size_t wave = 0; wave < plan.jobs.size();) {
            const size_t end =
                winnow::wave_end(plan.jobs, wave, shared_size, llama_n_ctx(context.get()), size_t(parallel));
            batch.n_tokens = 0;
            std::vector<size_t> outputs;
            auto flush = [&]() {
                decode();
                read_outputs(outputs, plan, counts, rows);
                outputs.clear();
                batch.n_tokens = 0;
            };
            for (size_t q = wave; q < end; ++q) {
                const int seq = int(q - wave + 2);
                llama_memory_seq_cp(memory, parent, seq, -1, -1);
                const auto &tokens = plan.jobs[q].tokens;
                leaf_tokens += tokens.size();
                for (size_t p = 0; p < tokens.size(); ++p) {
                    const bool last = p + 1 == tokens.size();
                    add(tokens[p], int(shared_size + p), seq, last);
                    if (last)
                        outputs.push_back(q);
                    if (batch.n_tokens == options.batch)
                        flush();
                }
            }
            if (batch.n_tokens)
                flush();
            for (size_t q = wave; q < end; ++q)
                if (!llama_memory_seq_rm(memory, int(q - wave + 2), -1, -1))
                    throw std::runtime_error("Could not release question cache");
            wave = end;
            ++waves;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        return {{"state_id", state_id},
                {"logits", rows},
                {"metrics",
                 {{"cache_hit", hit},
                  {"image_tokens", cached_image_tokens},
                  {"prefix_tokens", prefix_size},
                  {"prefix_reused_tokens", reused_prefix_tokens},
                  {"prefix_processed_tokens", prefix_size - reused_prefix_tokens},
                  {"request_prefix_tokens", common.size()},
                  {"request_prefix_reused_tokens", request_reused_tokens},
                  {"request_prefix_processed_tokens", common.size() - request_reused_tokens},
                  {"question_count", questions.size()},
                  {"unique_questions", plan.jobs.size()},
                  {"waves", waves},
                  {"suffix_tokens",
                   plan.original_suffix_tokens + (optimize ? request_prefix.size() * questions.size() : 0)},
                  {"leaf_processed_tokens", leaf_tokens},
                  {"token_cache_hits", token_hits},
                  {"token_cache_bytes", token_cache.bytes()},
                  {"tokenize_ms", tokenize_ms},
                  {"prefill_ms", prefill_ms},
                  {"request_prefill_ms", request_prefill_ms},
                  {"questions_ms", ms(t)},
                  {"native_ms", ms(start)},
                  {"decode_ms", decode_ms},
                  {"cooperative_yield_ms", yield_ms},
                  {"head", options.head},
                  {"pipeline", options.pipeline},
                  {"parallel", parallel},
                  {"backend_allocated_bytes", total - free},
                  {"backend_budget_bytes", total}}}};
    }
};

} // namespace winnow

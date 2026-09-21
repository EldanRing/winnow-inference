#include "bridge.h"
#include "engine.h"
#include "protocol.h"
#include <cstdlib>
#include <deque>

namespace winnow {
static int integer_env(const char *key, int fallback, int low, int high) {
    const char *value = std::getenv(key);
    if (!value)
        return fallback;
    size_t end = 0;
    int n = std::stoi(value, &end);
    if (end != std::string(value).size() || n < low || n > high)
        throw std::invalid_argument(std::string("Invalid ") + key);
    return n;
}
struct Service::Impl {
    llama_model *model;
    mtmd_context *vision;
    Options options;
    Hooks hooks;
    std::string model_name;
    std::unique_ptr<Engine> engine;
    std::deque<std::shared_ptr<Request>> queue;
    std::shared_ptr<Request> current;
    bool exclusive = false;
    size_t evictions = 0, cancellations = 0;
    Impl(llama_model *m, mtmd_context *v, int context, int batch, int ubatch, int threads, std::string name,
         Hooks h)
        : model(m), vision(v), hooks(std::move(h)), model_name(std::move(name)) {
        options.context = integer_env("WINNOW_CONTEXT", context, 512, llama_model_n_ctx_train(model));
        options.parallel = integer_env("WINNOW_PARALLEL", 1, 1, 32);
        options.batch = integer_env("WINNOW_BATCH", batch, 1, 65536);
        options.ubatch = integer_env("WINNOW_UBATCH", ubatch, 1, options.batch);
        options.threads = threads;
        if (const char *s = std::getenv("WINNOW_HEAD"))
            options.head = s;
        if (const char *s = std::getenv("WINNOW_CACHE"))
            options.cache = s;
        if (const char *s = std::getenv("WINNOW_PIPELINE"))
            options.pipeline = s;
        if ((options.head != "selected" && options.head != "full") ||
            (options.cache != "f16" && options.cache != "q8_0") ||
            (options.pipeline != "optimized" && options.pipeline != "reference"))
            throw std::invalid_argument("Invalid Winnow head, cache, or pipeline");
        if (const char *s = std::getenv("WINNOW_MEMORY")) {
            if (std::string(s) != "auto" && std::string(s) != "exclusive")
                throw std::invalid_argument("WINNOW_MEMORY must be auto or exclusive");
            exclusive = std::string(s) == "exclusive";
        }
    }
    void cooperate() {
        if (current && current->cancelled.load())
            throw std::runtime_error("Request cancelled");
        hooks.cooperate();
        if (current && current->cancelled.load())
            throw std::runtime_error("Request cancelled");
    }
    void create() {
        engine = std::make_unique<Engine>(options, model, vision, [this] { cooperate(); });
    }
};
Service::Service(llama_model *m, mtmd_context *v, int c, int b, int u, int t, const std::string &name,
                 Hooks hooks)
    : impl(new Impl(m, v, c, b, u, t, name, std::move(hooks))) {}
Service::~Service() {
    for (auto &r : impl->queue) {
        r->status = 503;
        r->done.set_value("{\"error\":{\"message\":\"Server shutting down\"}}");
    }
}
void Service::enqueue(std::shared_ptr<Request> r) {
    if (impl->queue.size() >= 128) {
        r->status = 429;
        r->done.set_value("{\"error\":{\"message\":\"Decision queue is full\"}}");
        return;
    }
    impl->queue.push_back(std::move(r));
}
bool Service::pending() const {
    return !impl->queue.empty();
}
bool Service::busy() const {
    return bool(impl->current);
}
void Service::release_idle() {
    if (busy())
        throw std::runtime_error("Cannot release an active decision context");
    if (impl->engine) {
        impl->engine.reset();
        ++impl->evictions;
    }
}
void Service::tick() {
    if (!pending())
        return;
    auto r = impl->queue.front();
    if (r->cancelled.load()) {
        ++impl->cancellations;
        impl->queue.pop_front();
        r->status = 499;
        r->done.set_value("{}");
        return;
    }
    if (!impl->engine && impl->exclusive && impl->hooks.chat_active())
        return;
    impl->queue.pop_front();
    impl->current = r;
    const double queue_ms = ms(r->received);
    try {
        // Validate JSON before allocating a decision context.
        auto input = json::parse(r->body);
        if (!input.is_object())
            throw std::invalid_argument("Expected an object");
        if (!impl->engine) {
            if (impl->exclusive)
                impl->hooks.suspend_chat();
            try {
                impl->create();
            } catch (const std::bad_alloc &) {
                impl->exclusive = true;
                if (impl->hooks.chat_active()) {
                    impl->queue.push_front(r);
                    impl->current.reset();
                    return;
                }
                impl->hooks.suspend_chat();
                impl->create();
            }
        }
        auto info = impl->engine->info();
        auto compiled = compile(input, info["labels"].get<std::vector<std::string>>(),
                                info["chat_template"].get<std::string>());
        if (input.contains("model") && input["model"] != impl->model_name && input["model"] != "Winnow-12B" &&
            input["model"] != "jev-latest")
            throw std::invalid_argument("Unknown model; use Winnow-12B or the configured alias");
        json out;
        if (r->inspect) {
            out = impl->engine->inspect(compiled.native);
            out["runtime"] = info;
        } else {
            const auto raw = impl->engine->evaluate(compiled.native);
            out = {{"model", impl->model_name}, {"answers", json::object()}};
            for (size_t i = 0; i < compiled.questions.size(); ++i)
                out["answers"][compiled.questions[i].id] =
                    answer(compiled.questions[i], raw["logits"][i].get<std::vector<float>>(),
                           compiled.temperature, compiled.diagnostics);
            const auto &metrics = raw["metrics"];
            out["usage"] = {{"input_tokens",
                             metrics["prefix_tokens"].get<size_t>() + metrics["suffix_tokens"].get<size_t>()},
                            {"output_tokens", 0}};
            if (compiled.diagnostics) {
                out["winnow"] = metrics;
                out["winnow"]["queue_ms"] = queue_ms;
                out["winnow"]["context_evictions"] = impl->evictions;
                out["winnow"]["cancelled_requests"] = impl->cancellations;
                out["winnow"]["memory_policy"] = impl->exclusive ? "exclusive" : "mixed";
                out["winnow"]["context_capacity"] = impl->options.context;
            }
        }
        r->done.set_value(out.dump());
    } catch (const json::exception &e) {
        r->status = 400;
        r->done.set_value(
            json({{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}).dump());
    } catch (const std::invalid_argument &e) {
        r->status = 400;
        r->done.set_value(
            json({{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}).dump());
    } catch (const std::exception &e) {
        if (impl->engine)
            impl->engine->reset();
        if (r->cancelled.load())
            ++impl->cancellations;
        r->status = r->cancelled.load() ? 499 : 500;
        r->done.set_value(json({{"error", {{"message", e.what()}, {"type", "inference_error"}}}}).dump());
    }
    impl->current.reset();
}
} // namespace winnow

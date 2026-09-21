#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <string>

struct llama_model;
struct mtmd_context;
namespace winnow {
struct Request {
    std::chrono::steady_clock::time_point received = std::chrono::steady_clock::now();
    std::string body;
    bool inspect = false;
    std::atomic<bool> cancelled{false};
    std::promise<std::string> done;
    int status = 200;
};
struct Hooks {
    // All callbacks run on llama-server's inference thread.
    std::function<bool()> chat_active;
    std::function<void()> suspend_chat;
    std::function<void()> cooperate;
};
class Service {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    Service(llama_model *, mtmd_context *, int context, int batch, int ubatch, int threads,
            const std::string &model_name, Hooks);
    ~Service();
    void enqueue(std::shared_ptr<Request>);
    bool pending() const;
    void tick();
    // Called before recreating chat context; model weights remain resident.
    void release_idle();
    bool busy() const;
};
} // namespace winnow

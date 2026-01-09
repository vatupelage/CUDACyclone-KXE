// ============================================================================
// CUDACyclone Distributed Mode - Server Header
// ============================================================================
// Coordinates distributed work across multiple client machines
// ============================================================================

#ifndef CUDACYCLONE_SERVER_H
#define CUDACYCLONE_SERVER_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>

#include "CUDACyclone_Protocol.h"
#include "CUDACyclone_Network.h"
#include "CUDACyclone_WorkUnit.h"

// Checkpoint constants
constexpr uint64_t CHECKPOINT_MAGIC = 0x4353525653434B50ULL;  // "CSRVSCKP"
constexpr uint32_t CHECKPOINT_VERSION = 1;

// ============================================================================
// SERVER CONFIGURATION
// ============================================================================

struct ServerConfig {
    // Search parameters
    uint64_t range_start[4];            // Search range start (256-bit)
    uint64_t range_end[4];              // Search range end (256-bit)
    uint8_t target_hash160[20];         // Target address hash

    // Work distribution
    uint32_t work_unit_bits;            // Log2 of work unit size
    uint32_t batch_size;                // Recommended batch size for clients
    uint32_t slices_per_launch;         // Recommended slices for clients
    bool pincer_mode;                   // Enable bidirectional mode

    // KXE mode
    bool kxe_mode;                      // Enable KXE permuted scanning
    uint64_t kxe_seed;                  // KXE permutation seed

    // Network
    uint16_t port;                      // Server port
    uint32_t max_clients;               // Maximum concurrent clients

    // Timing
    uint32_t heartbeat_interval_sec;    // Expected heartbeat interval
    uint32_t heartbeat_timeout_sec;     // Timeout before reassignment
    uint32_t progress_interval_sec;     // Expected progress report interval
    uint32_t checkpoint_interval_sec;   // Server state checkpoint interval

    // Resilience
    uint32_t max_reassign_count;        // Max reassigns before permanent failure
    std::string checkpoint_file;        // Path to checkpoint file

    ServerConfig() {
        arith256::zero(range_start);
        arith256::zero(range_end);
        memset(target_hash160, 0, sizeof(target_hash160));
        work_unit_bits = DEFAULT_WORK_UNIT_BITS;
        batch_size = 128;
        slices_per_launch = 16;
        pincer_mode = false;
        kxe_mode = false;
        kxe_seed = 0;
        port = DEFAULT_SERVER_PORT;
        max_clients = MAX_CLIENTS;
        heartbeat_interval_sec = DEFAULT_HEARTBEAT_INTERVAL_SEC;
        heartbeat_timeout_sec = DEFAULT_HEARTBEAT_TIMEOUT_SEC;
        progress_interval_sec = DEFAULT_PROGRESS_INTERVAL_SEC;
        checkpoint_interval_sec = 300;  // 5 minutes
        max_reassign_count = MAX_REASSIGN_COUNT;
    }
};

// ============================================================================
// CLIENT INFORMATION
// ============================================================================

struct ClientInfo {
    uint32_t client_id;                 // Unique client ID
    socket_t socket;                    // Client socket
    std::string hostname;               // Client hostname
    std::string ip_address;             // Client IP
    uint16_t port;                      // Client port

    // Hardware
    uint32_t gpu_count;                 // Number of GPUs
    std::vector<GPUInfoMsg> gpus;       // GPU information
    bool supports_pincer;               // Supports bidirectional mode

    // Performance
    double current_speed_gkeys;         // Current aggregate speed
    double peak_speed_gkeys;            // Peak speed observed
    uint64_t total_keys_processed;      // Total keys processed

    // State
    std::vector<uint32_t> active_units; // Currently assigned unit IDs
    uint64_t last_heartbeat;            // Last heartbeat timestamp
    uint64_t last_progress;             // Last progress report timestamp
    uint64_t connected_at;              // Connection timestamp
    bool connected;                     // Currently connected

    std::thread handler_thread;         // Client handler thread

    ClientInfo() {
        client_id = 0;
        socket = INVALID_SOCKET_VALUE;
        port = 0;
        gpu_count = 0;
        supports_pincer = false;
        current_speed_gkeys = 0.0;
        peak_speed_gkeys = 0.0;
        total_keys_processed = 0;
        last_heartbeat = 0;
        last_progress = 0;
        connected_at = 0;
        connected = false;
    }
};

// ============================================================================
// SERVER CHECKPOINT FORMAT
// ============================================================================

#pragma pack(push, 1)

struct ServerCheckpointHeader {
    uint64_t magic;                     // Checkpoint magic number
    uint32_t version;                   // Checkpoint version
    uint64_t timestamp;                 // Save timestamp
    uint64_t range_start[4];            // Original range
    uint64_t range_end[4];
    uint8_t target_hash160[20];         // Target address
    uint32_t work_unit_bits;            // Unit size
    uint32_t total_units;               // Total work units
    uint32_t completed_units;           // Completed count
    uint64_t total_keys_processed;      // Aggregate keys
    uint32_t batch_size;
    uint32_t slices;
    uint8_t pincer_mode;
    uint8_t found;                      // 1 if key found
    uint8_t kxe_mode;                   // 1 if KXE mode
    uint8_t reserved;
    uint64_t kxe_seed;                  // KXE seed (if KXE mode)
    uint64_t found_scalar[4];           // Found key (if any)
    // Followed by: WorkUnit data array
};

#pragma pack(pop)

// ============================================================================
// SERVER CLASS
// ============================================================================

class CycloneServer {
public:
    CycloneServer(const ServerConfig& config);
    ~CycloneServer();

    // Lifecycle
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Wait for completion
    void wait_for_completion();

    // Manual checkpoint
    bool save_checkpoint();
    bool load_checkpoint();

    // Status
    void print_status() const;
    uint64_t get_total_keys_processed() const;
    double get_aggregate_speed() const;

private:
    // Thread functions
    void accept_thread();
    void client_handler(uint32_t client_id);
    void maintenance_thread();
    void status_thread();

    // Message handlers
    bool handle_register(uint32_t client_id, const void* payload, uint16_t size);
    bool handle_heartbeat(uint32_t client_id, const HeartbeatMsg& msg);
    bool handle_work_request(uint32_t client_id, const WorkRequestMsg& msg);
    bool handle_progress_report(uint32_t client_id, const ProgressReportMsg& msg);
    bool handle_unit_complete(uint32_t client_id, const UnitCompleteMsg& msg);
    bool handle_found_result(uint32_t client_id, const FoundResultMsg& msg);
    bool handle_disconnect(uint32_t client_id);

    // Work distribution
    WorkUnit* assign_work_unit(uint32_t client_id);
    void release_work_unit(uint32_t unit_id);
    bool send_work_assignment(uint32_t client_id, const WorkUnit& wu);
    bool send_no_work_available(uint32_t client_id, uint32_t reason);

    // Client management
    uint32_t allocate_client_id();
    void disconnect_client(uint32_t client_id);
    void broadcast_key_found(const uint64_t scalar[4], uint32_t finder_client_id, uint32_t unit_id);
    void broadcast_shutdown(uint32_t reason);

    // Result verification
    bool verify_result(const FoundResultMsg& result);

    // Utilities
    uint64_t current_time_sec() const;
    std::string format_time(uint64_t seconds) const;
    std::string format_speed(double gkeys) const;
    std::string format_keys(uint64_t keys) const;

    // Configuration
    ServerConfig config_;

    // Work management
    WorkUnitManager work_manager_;
    std::mutex work_mutex_;

    // Network
    socket_t server_socket_;

    // Client management
    std::map<uint32_t, ClientInfo> clients_;
    std::mutex clients_mutex_;
    uint32_t next_client_id_;
    std::thread accept_thread_;

    // Maintenance
    std::thread maintenance_thread_;
    std::thread status_thread_;

    // State
    std::atomic<bool> running_{false};
    std::atomic<bool> found_{false};
    uint64_t found_scalar_[4];
    uint32_t found_by_client_;
    uint32_t found_unit_id_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_checkpoint_time_;

    // Synchronization
    std::condition_variable completion_cv_;
    std::mutex completion_mutex_;
};

// ============================================================================
// COMMAND-LINE PARSING HELPERS
// ============================================================================

// Parse server command-line arguments
bool parse_server_args(int argc, char* argv[], ServerConfig& config);

// Print server usage
void print_server_usage(const char* prog_name);

#endif // CUDACYCLONE_SERVER_H

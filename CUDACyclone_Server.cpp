// ============================================================================
// CUDACyclone Distributed Mode - Server Implementation
// ============================================================================

#include "CUDACyclone_Server.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <csignal>
#include <ctime>
#include <vector>
#include <random>

// Global server pointer for signal handling
static CycloneServer* g_server = nullptr;
static std::atomic<bool> g_shutdown_requested{false};

static void signal_handler(int sig) {
    (void)sig;
    g_shutdown_requested.store(true);
    if (g_server) {
        g_server->stop();
    }
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

CycloneServer::CycloneServer(const ServerConfig& config)
    : config_(config),
      server_socket_(INVALID_SOCKET_VALUE),
      next_client_id_(1),
      found_by_client_(0),
      found_unit_id_(0) {
    arith256::zero(found_scalar_);

    // Register signal handlers
    g_server = this;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
}

CycloneServer::~CycloneServer() {
    stop();
    g_server = nullptr;
}

// ============================================================================
// LIFECYCLE
// ============================================================================

bool CycloneServer::start() {
    if (running_.load()) {
        std::cerr << "[Server] Already running\n";
        return false;
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              CUDACyclone Distributed Server v1.0                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Initialize network
    if (!net::initialize()) {
        std::cerr << "[Server] Failed to initialize network\n";
        return false;
    }

    // Initialize work units
    std::cout << "[Server] Initializing work units...\n";
    if (!work_manager_.initialize(config_.range_start, config_.range_end, config_.work_unit_bits)) {
        std::cerr << "[Server] Failed to initialize work units\n";
        std::cerr << "[Server] Ensure range is power-of-2 and properly aligned\n";
        return false;
    }

    // Generate random KXE seed if not provided
    if (config_.kxe_mode && config_.kxe_seed == 0) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        config_.kxe_seed = gen();
    }

    std::cout << "[Server] Range: " << arith256::to_hex(config_.range_start)
              << " : " << arith256::to_hex(config_.range_end) << "\n";
    std::cout << "[Server] Work unit size: 2^" << config_.work_unit_bits << " keys\n";
    std::cout << "[Server] Total work units: " << work_manager_.get_total_units() << "\n";
    std::cout << "[Server] Pincer mode: " << (config_.pincer_mode ? "enabled" : "disabled") << "\n";
    std::cout << "[Server] KXE mode: " << (config_.kxe_mode ? "enabled" : "disabled");
    if (config_.kxe_mode) {
        std::cout << " (seed: " << config_.kxe_seed << ")";
    }
    std::cout << "\n";

    // Try to load checkpoint
    if (!config_.checkpoint_file.empty()) {
        if (load_checkpoint()) {
            std::cout << "[Server] Resumed from checkpoint\n";
        }
    }

    // Create server socket
    std::cout << "[Server] Starting on port " << config_.port << "...\n";
    server_socket_ = net::create_server_socket(config_.port, 10);
    if (server_socket_ == INVALID_SOCKET_VALUE) {
        std::cerr << "[Server] Failed to create server socket: " << net::get_last_error_string() << "\n";
        return false;
    }

    running_.store(true);
    start_time_ = std::chrono::steady_clock::now();
    last_checkpoint_time_ = start_time_;

    // Start threads
    accept_thread_ = std::thread(&CycloneServer::accept_thread, this);
    maintenance_thread_ = std::thread(&CycloneServer::maintenance_thread, this);
    status_thread_ = std::thread(&CycloneServer::status_thread, this);

    std::cout << "[Server] Ready for connections\n\n";

    return true;
}

void CycloneServer::stop() {
    if (!running_.exchange(false)) {
        return;  // Already stopped
    }

    std::cout << "\n[Server] Shutting down...\n";

    // Broadcast shutdown to all clients
    broadcast_shutdown(0);

    // Close server socket to unblock accept
    if (server_socket_ != INVALID_SOCKET_VALUE) {
        net::close_socket(server_socket_);
        server_socket_ = INVALID_SOCKET_VALUE;
    }

    // Wait for accept thread
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    // Disconnect all clients and collect threads to join
    std::vector<std::thread> threads_to_join;
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto& [id, client] : clients_) {
            if (client.socket != INVALID_SOCKET_VALUE) {
                net::close_socket(client.socket);
                client.socket = INVALID_SOCKET_VALUE;
            }
            client.connected = false;
            if (client.handler_thread.joinable()) {
                threads_to_join.push_back(std::move(client.handler_thread));
            }
        }
    }

    // Join threads outside the lock to avoid deadlock
    for (auto& t : threads_to_join) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Signal maintenance and status threads
    completion_cv_.notify_all();

    if (maintenance_thread_.joinable()) {
        maintenance_thread_.join();
    }
    if (status_thread_.joinable()) {
        status_thread_.join();
    }

    // Save final checkpoint
    if (!config_.checkpoint_file.empty()) {
        save_checkpoint();
    }

    net::cleanup();

    std::cout << "[Server] Shutdown complete\n";
}

void CycloneServer::wait_for_completion() {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this] {
        return !running_.load() || found_.load() || work_manager_.is_all_completed();
    });
}

// ============================================================================
// THREAD FUNCTIONS
// ============================================================================

void CycloneServer::accept_thread() {
    while (running_.load()) {
        std::string client_addr;
        uint16_t client_port;

        socket_t client_socket = net::accept_connection(
            server_socket_, client_addr, client_port, 1000);

        if (client_socket == INVALID_SOCKET_VALUE) {
            if (!running_.load()) break;
            continue;  // Timeout or error, retry
        }

        // Check client limit
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            if (clients_.size() >= config_.max_clients) {
                std::cout << "[Server] Rejecting connection from " << client_addr
                          << " (max clients reached)\n";

                ErrorResponseMsg err;
                err.error_code = static_cast<uint32_t>(ErrorCode::SERVER_FULL);
                net::send_message(client_socket, MessageType::ERROR_RESPONSE,
                                  &err, sizeof(err));
                net::close_socket(client_socket);
                continue;
            }
        }

        // Allocate client ID and create entry
        uint32_t client_id = allocate_client_id();

        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            ClientInfo& client = clients_[client_id];
            client.client_id = client_id;
            client.socket = client_socket;
            client.ip_address = client_addr;
            client.port = client_port;
            client.connected_at = current_time_sec();
            client.last_heartbeat = client.connected_at;
            client.connected = true;

            std::cout << "[Server] New connection from " << client_addr << ":"
                      << client_port << " (client #" << client_id << ")\n" << std::flush;

            // Start handler thread
            client.handler_thread = std::thread(&CycloneServer::client_handler, this, client_id);
        }
    }
}

void CycloneServer::client_handler(uint32_t client_id) {
    socket_t sock = INVALID_SOCKET_VALUE;

    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = clients_.find(client_id);
        if (it == clients_.end()) return;
        sock = it->second.socket;
    }

    // Message receive buffer
    std::vector<uint8_t> buffer(4096);

    while (running_.load()) {
        // Receive message header
        MessageHeader header;
        if (!net::recv_message_header(sock, header, 1000)) {
            if (!running_.load()) break;

            // Check if socket is still valid
            int poll_result = net::poll_readable(sock, 0);
            if (poll_result < 0) {
                // Socket error, disconnect
                break;
            }
            continue;  // Timeout, keep waiting
        }

        // Receive payload if any
        if (header.payload_size > 0) {
            if (header.payload_size > buffer.size()) {
                buffer.resize(header.payload_size);
            }
            if (!net::recv_message_payload(sock, buffer.data(), header.payload_size)) {
                break;  // Disconnect
            }
        }

        // Process message
        bool success = true;
        switch (header.type()) {
            case MessageType::REGISTER_REQUEST:
                success = handle_register(client_id, buffer.data(), header.payload_size);
                break;

            case MessageType::HEARTBEAT:
                if (header.payload_size >= sizeof(HeartbeatMsg)) {
                    success = handle_heartbeat(client_id,
                        *reinterpret_cast<HeartbeatMsg*>(buffer.data()));
                }
                break;

            case MessageType::WORK_REQUEST:
                if (header.payload_size >= sizeof(WorkRequestMsg)) {
                    success = handle_work_request(client_id,
                        *reinterpret_cast<WorkRequestMsg*>(buffer.data()));
                }
                break;

            case MessageType::PROGRESS_REPORT:
                if (header.payload_size >= sizeof(ProgressReportMsg)) {
                    success = handle_progress_report(client_id,
                        *reinterpret_cast<ProgressReportMsg*>(buffer.data()));
                }
                break;

            case MessageType::UNIT_COMPLETE:
                if (header.payload_size >= sizeof(UnitCompleteMsg)) {
                    success = handle_unit_complete(client_id,
                        *reinterpret_cast<UnitCompleteMsg*>(buffer.data()));
                }
                break;

            case MessageType::FOUND_RESULT:
                if (header.payload_size >= sizeof(FoundResultMsg)) {
                    success = handle_found_result(client_id,
                        *reinterpret_cast<FoundResultMsg*>(buffer.data()));
                }
                break;

            case MessageType::DISCONNECT:
                success = handle_disconnect(client_id);
                break;

            default:
                std::cerr << "[Server] Unknown message type from client #" << client_id
                          << ": 0x" << std::hex << header.msg_type << std::dec << "\n";
                break;
        }

        if (!success) {
            break;  // Handler indicated disconnect
        }
    }

    // Client disconnected
    disconnect_client(client_id);
}

void CycloneServer::maintenance_thread() {
    while (running_.load()) {
        // Sleep for a bit
        std::this_thread::sleep_for(std::chrono::seconds(5));

        if (!running_.load()) break;

        uint64_t now = current_time_sec();

        // Expire stale work units
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            work_manager_.expire_stale(config_.heartbeat_timeout_sec, now);
            work_manager_.reset_expired(config_.max_reassign_count);
        }

        // Check for completed clients (no heartbeat)
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            for (auto& [id, client] : clients_) {
                if (client.connected &&
                    now - client.last_heartbeat > config_.heartbeat_timeout_sec) {
                    std::cout << "[Server] Client #" << id << " timed out\n";
                    // Will be cleaned up by handler thread exit
                }
            }
        }

        // Periodic checkpoint
        auto now_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now_time - last_checkpoint_time_).count();
        if (!config_.checkpoint_file.empty() &&
            elapsed >= config_.checkpoint_interval_sec) {
            save_checkpoint();
            last_checkpoint_time_ = now_time;
        }

        // Check for completion
        if (work_manager_.is_all_completed()) {
            std::cout << "\n[Server] All work units completed!\n";
            completion_cv_.notify_all();
            if (!found_.load()) {
                std::cout << "[Server] Key NOT found in searched range.\n";
            }
            stop();
            break;
        }
    }
}

void CycloneServer::status_thread() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!running_.load()) break;

        print_status();
    }
}

// ============================================================================
// MESSAGE HANDLERS
// ============================================================================

bool CycloneServer::handle_register(uint32_t client_id, const void* payload, uint16_t size) {
    if (size < sizeof(RegisterRequestMsg)) {
        return false;
    }

    const auto* msg = static_cast<const RegisterRequestMsg*>(payload);
    const uint8_t* data = static_cast<const uint8_t*>(payload);

    // Check protocol version
    if (msg->protocol_version != PROTOCOL_VERSION) {
        RegisterResponseMsg resp;
        resp.status = static_cast<uint32_t>(ErrorCode::VERSION_MISMATCH);
        resp.client_id = 0;

        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = clients_.find(client_id);
        if (it != clients_.end()) {
            net::send_message(it->second.socket, MessageType::REGISTER_RESPONSE,
                              &resp, sizeof(resp));
        }
        return false;
    }

    // Parse variable-length parts
    size_t offset = sizeof(RegisterRequestMsg);
    std::string hostname;
    if (msg->hostname_len > 0 && offset + msg->hostname_len <= size) {
        hostname = std::string(reinterpret_cast<const char*>(data + offset), msg->hostname_len);
        offset += msg->hostname_len;
    }

    std::vector<GPUInfoMsg> gpus;
    for (uint32_t i = 0; i < msg->gpu_count && offset + sizeof(GPUInfoMsg) <= size; ++i) {
        gpus.push_back(*reinterpret_cast<const GPUInfoMsg*>(data + offset));
        offset += sizeof(GPUInfoMsg);
    }

    // Update client info
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = clients_.find(client_id);
        if (it == clients_.end()) return false;

        ClientInfo& client = it->second;
        client.hostname = hostname.empty() ? client.ip_address : hostname;
        client.gpu_count = msg->gpu_count;
        client.gpus = std::move(gpus);
        client.supports_pincer = msg->supports_pincer != 0;

        std::cout << "[Server] Client #" << client_id << " registered: "
                  << client.hostname << " with " << client.gpu_count << " GPU(s)";
        if (!client.gpus.empty()) {
            std::cout << " [" << client.gpus[0].name << "]";
        }
        std::cout << "\n";

        // Send response
        RegisterResponseMsg resp;
        resp.status = static_cast<uint32_t>(ErrorCode::OK);
        resp.client_id = client_id;
        resp.heartbeat_interval = config_.heartbeat_interval_sec;
        resp.progress_interval = config_.progress_interval_sec;
        resp.max_units_per_request = MAX_WORK_UNITS_PER_REQUEST;
        resp.scan_mode = config_.kxe_mode ? static_cast<uint8_t>(ScanMode::KXE)
                                          : static_cast<uint8_t>(ScanMode::SEQUENTIAL);
        resp.kxe_seed = config_.kxe_seed;
        resp.total_blocks = work_manager_.get_total_units();

        net::send_message(client.socket, MessageType::REGISTER_RESPONSE, &resp, sizeof(resp));
    }

    return true;
}

bool CycloneServer::handle_heartbeat(uint32_t client_id, const HeartbeatMsg& msg) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = clients_.find(client_id);
    if (it == clients_.end()) return false;

    ClientInfo& client = it->second;
    client.last_heartbeat = current_time_sec();
    client.current_speed_gkeys = msg.current_speed_gkeys;
    if (msg.current_speed_gkeys > client.peak_speed_gkeys) {
        client.peak_speed_gkeys = msg.current_speed_gkeys;
    }

    // Send acknowledgment
    HeartbeatAckMsg ack;
    ack.status = static_cast<uint32_t>(ErrorCode::OK);
    ack.server_time = static_cast<uint32_t>(current_time_sec());

    net::send_message(client.socket, MessageType::HEARTBEAT_ACK, &ack, sizeof(ack));

    return true;
}

bool CycloneServer::handle_work_request(uint32_t client_id, const WorkRequestMsg& msg) {
    (void)msg;  // units_requested currently ignored, assign one at a time

    // Check if key already found
    if (found_.load()) {
        send_no_work_available(client_id, 1);  // Search complete
        return true;
    }

    // Try to assign a work unit
    WorkUnit* wu = assign_work_unit(client_id);

    if (wu) {
        // Record assignment
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            auto it = clients_.find(client_id);
            if (it != clients_.end()) {
                it->second.active_units.push_back(wu->unit_id);
            }
        }

        send_work_assignment(client_id, *wu);
    } else {
        // No work available
        uint32_t reason = work_manager_.is_all_completed() ? 1 : 0;
        send_no_work_available(client_id, reason);
    }

    return true;
}

bool CycloneServer::handle_progress_report(uint32_t client_id, const ProgressReportMsg& msg) {
    std::lock_guard<std::mutex> lock(work_mutex_);

    WorkUnit* wu = work_manager_.get_unit(msg.unit_id);
    if (!wu || wu->assigned_client_id != client_id) {
        return true;  // Ignore stale report
    }

    wu->keys_processed = msg.keys_processed;

    // Update client stats
    {
        std::lock_guard<std::mutex> lock2(clients_mutex_);
        auto it = clients_.find(client_id);
        if (it != clients_.end()) {
            it->second.last_progress = current_time_sec();
            it->second.current_speed_gkeys = msg.speed_gkeys;
        }
    }

    return true;
}

bool CycloneServer::handle_unit_complete(uint32_t client_id, const UnitCompleteMsg& msg) {
    {
        std::lock_guard<std::mutex> lock(work_mutex_);

        WorkUnit* wu = work_manager_.get_unit(msg.unit_id);
        if (!wu) return true;

        wu->state = WorkUnitState::COMPLETED;
        wu->completed_at = current_time_sec();
        wu->keys_processed = msg.keys_processed;

        std::cout << "[Server] Unit #" << msg.unit_id << " completed by client #"
                  << client_id << " (" << format_speed(msg.avg_speed_gkeys) << " avg)\n";
    }

    // Update client stats
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = clients_.find(client_id);
        if (it != clients_.end()) {
            it->second.total_keys_processed += msg.keys_processed;
            auto& units = it->second.active_units;
            units.erase(std::remove(units.begin(), units.end(), msg.unit_id), units.end());
        }
    }

    return true;
}

bool CycloneServer::handle_found_result(uint32_t client_id, const FoundResultMsg& msg) {
    // Verify result
    if (!verify_result(msg)) {
        std::cerr << "[Server] Invalid result from client #" << client_id << " - verification failed!\n";

        ErrorResponseMsg err;
        err.error_code = static_cast<uint32_t>(ErrorCode::INVALID_RESULT);

        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = clients_.find(client_id);
        if (it != clients_.end()) {
            net::send_message(it->second.socket, MessageType::ERROR_RESPONSE, &err, sizeof(err));
        }
        return true;
    }

    // Record the find
    if (!found_.exchange(true)) {
        arith256::copy(msg.scalar, found_scalar_);
        found_by_client_ = client_id;
        found_unit_id_ = msg.unit_id;

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                        KEY FOUND!                                ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        std::cout << "  Private Key: " << arith256::to_hex(msg.scalar) << "\n";
        std::cout << "  Found by: Client #" << client_id << " (unit #" << msg.unit_id << ")\n";
        std::cout << "\n";

        // Broadcast to all clients
        broadcast_key_found(msg.scalar, client_id, msg.unit_id);

        // Save checkpoint
        if (!config_.checkpoint_file.empty()) {
            save_checkpoint();
        }

        // Signal completion
        completion_cv_.notify_all();
    }

    return true;
}

bool CycloneServer::handle_disconnect(uint32_t client_id) {
    std::cout << "[Server] Client #" << client_id << " disconnecting\n";
    return false;  // Signal to close connection
}

// ============================================================================
// WORK DISTRIBUTION
// ============================================================================

WorkUnit* CycloneServer::assign_work_unit(uint32_t client_id) {
    std::lock_guard<std::mutex> lock(work_mutex_);

    // Check client support for pincer mode
    bool supports_pincer = false;
    {
        std::lock_guard<std::mutex> lock2(clients_mutex_);
        auto it = clients_.find(client_id);
        if (it != clients_.end()) {
            supports_pincer = it->second.supports_pincer && config_.pincer_mode;
        }
    }

    // Get next available unit
    WorkUnit* wu = config_.pincer_mode ?
        work_manager_.get_next_available_pincer_aware(supports_pincer) :
        work_manager_.get_next_available();

    if (!wu) return nullptr;

    // Assign to client
    wu->state = WorkUnitState::ASSIGNED;
    wu->assigned_client_id = client_id;
    wu->assigned_at = current_time_sec();

    return wu;
}

void CycloneServer::release_work_unit(uint32_t unit_id) {
    std::lock_guard<std::mutex> lock(work_mutex_);

    WorkUnit* wu = work_manager_.get_unit(unit_id);
    if (wu && wu->state == WorkUnitState::ASSIGNED) {
        wu->state = WorkUnitState::AVAILABLE;
        wu->assigned_client_id = 0;
        wu->assigned_at = 0;
    }
}

bool CycloneServer::send_work_assignment(uint32_t client_id, const WorkUnit& wu) {
    WorkAssignmentMsg msg;
    msg.unit_id = wu.unit_id;
    // For KXE mode, send global range (client computes local range from block index)
    // For sequential mode, send per-unit range
    arith256::copy(config_.kxe_mode ? config_.range_start : wu.range_start, msg.range_start);
    arith256::copy(config_.kxe_mode ? config_.range_end : wu.range_end, msg.range_end);
    memcpy(msg.target_hash160, config_.target_hash160, 20);
    msg.batch_size = config_.batch_size;
    msg.slices = config_.slices_per_launch;
    msg.scan_direction = static_cast<uint8_t>(wu.direction);
    msg.pincer_enabled = config_.pincer_mode ? 1 : 0;
    msg.pincer_partner_unit = wu.pincer_partner_id;

    // KXE mode fields
    msg.scan_mode = config_.kxe_mode ? static_cast<uint8_t>(ScanMode::KXE)
                                     : static_cast<uint8_t>(ScanMode::SEQUENTIAL);
    msg.kxe_block_index = wu.unit_id;  // In KXE mode, unit_id is the block index
    msg.kxe_seed = config_.kxe_seed;

    // Calculate keys_per_block from work unit size
    uint64_t keys_per_block = 1ULL << config_.work_unit_bits;
    msg.keys_per_block = keys_per_block;

    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = clients_.find(client_id);
    if (it == clients_.end()) return false;

    return net::send_message(it->second.socket, MessageType::WORK_ASSIGNMENT, &msg, sizeof(msg));
}

bool CycloneServer::send_no_work_available(uint32_t client_id, uint32_t reason) {
    NoWorkAvailableMsg msg;
    msg.reason = reason;
    msg.retry_after_sec = (reason == 0) ? 5 : 0;

    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = clients_.find(client_id);
    if (it == clients_.end()) return false;

    return net::send_message(it->second.socket, MessageType::NO_WORK_AVAILABLE, &msg, sizeof(msg));
}

// ============================================================================
// CLIENT MANAGEMENT
// ============================================================================

uint32_t CycloneServer::allocate_client_id() {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return next_client_id_++;
}

void CycloneServer::disconnect_client(uint32_t client_id) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = clients_.find(client_id);
    if (it == clients_.end()) return;

    ClientInfo& client = it->second;
    client.connected = false;

    // Close socket
    if (client.socket != INVALID_SOCKET_VALUE) {
        net::close_socket(client.socket);
        client.socket = INVALID_SOCKET_VALUE;
    }

    // Release assigned work units
    for (uint32_t unit_id : client.active_units) {
        release_work_unit(unit_id);
    }
    client.active_units.clear();

    std::cout << "[Server] Client #" << client_id << " (" << client.hostname
              << ") disconnected\n";
}

void CycloneServer::broadcast_key_found(const uint64_t scalar[4], uint32_t finder_client_id, uint32_t unit_id) {
    KeyFoundMsg msg;
    msg.finder_client_id = finder_client_id;
    msg.unit_id = unit_id;
    arith256::copy(scalar, msg.scalar);

    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    msg.search_time_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto& [id, client] : clients_) {
        if (client.connected && client.socket != INVALID_SOCKET_VALUE) {
            net::send_message(client.socket, MessageType::KEY_FOUND, &msg, sizeof(msg));
        }
    }
}

void CycloneServer::broadcast_shutdown(uint32_t reason) {
    ServerShutdownMsg msg;
    msg.reason = reason;

    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto& [id, client] : clients_) {
        if (client.connected && client.socket != INVALID_SOCKET_VALUE) {
            net::send_message(client.socket, MessageType::SERVER_SHUTDOWN, &msg, sizeof(msg));
        }
    }
}

// ============================================================================
// RESULT VERIFICATION
// ============================================================================

bool CycloneServer::verify_result(const FoundResultMsg& result) {
    // TODO: Implement actual verification
    // 1. Compute public key from scalar
    // 2. Hash public key to get Hash160
    // 3. Compare with target_hash160
    // For now, just check that hash160 matches target
    return memcmp(result.hash160, config_.target_hash160, 20) == 0;
}

// ============================================================================
// CHECKPOINT
// ============================================================================

bool CycloneServer::save_checkpoint() {
    std::string temp_file = config_.checkpoint_file + ".tmp";
    std::ofstream ofs(temp_file, std::ios::binary);
    if (!ofs) {
        std::cerr << "[Server] Failed to create checkpoint file\n";
        return false;
    }

    // Build header
    ServerCheckpointHeader header;
    header.magic = CHECKPOINT_MAGIC;
    header.version = 2;  // v2 adds KXE mode
    header.timestamp = current_time_sec();
    arith256::copy(config_.range_start, header.range_start);
    arith256::copy(config_.range_end, header.range_end);
    memcpy(header.target_hash160, config_.target_hash160, 20);
    header.work_unit_bits = config_.work_unit_bits;
    header.total_units = work_manager_.get_total_units();
    header.completed_units = work_manager_.count_by_state(WorkUnitState::COMPLETED);
    header.total_keys_processed = get_total_keys_processed();
    header.batch_size = config_.batch_size;
    header.slices = config_.slices_per_launch;
    header.pincer_mode = config_.pincer_mode ? 1 : 0;
    header.found = found_.load() ? 1 : 0;
    header.kxe_mode = config_.kxe_mode ? 1 : 0;
    header.reserved = 0;
    header.kxe_seed = config_.kxe_seed;
    arith256::copy(found_scalar_, header.found_scalar);

    // Write header
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write work unit states
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        for (const auto& wu : work_manager_.get_units()) {
            uint32_t state_data[4];
            state_data[0] = wu.unit_id;
            state_data[1] = static_cast<uint32_t>(wu.state);
            state_data[2] = wu.reassign_count;
            state_data[3] = 0;

            ofs.write(reinterpret_cast<const char*>(state_data), sizeof(state_data));
            ofs.write(reinterpret_cast<const char*>(&wu.keys_processed), sizeof(wu.keys_processed));
        }
    }

    ofs.close();

    // Atomic rename
    std::rename(temp_file.c_str(), config_.checkpoint_file.c_str());

    std::cout << "[Server] Checkpoint saved (" << header.completed_units << "/"
              << header.total_units << " units completed)\n";

    return true;
}

bool CycloneServer::load_checkpoint() {
    std::ifstream ifs(config_.checkpoint_file, std::ios::binary);
    if (!ifs) {
        return false;  // No checkpoint file
    }

    ServerCheckpointHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic != CHECKPOINT_MAGIC) {
        std::cerr << "[Server] Invalid checkpoint file\n";
        return false;
    }

    // Validate parameters match
    if (arith256::compare(header.range_start, config_.range_start) != 0 ||
        arith256::compare(header.range_end, config_.range_end) != 0 ||
        memcmp(header.target_hash160, config_.target_hash160, 20) != 0) {
        std::cerr << "[Server] Checkpoint parameters don't match current config\n";
        return false;
    }

    // Load work unit states
    std::lock_guard<std::mutex> lock(work_mutex_);
    for (auto& wu : work_manager_.get_units()) {
        uint32_t state_data[4];
        uint64_t keys_processed;

        ifs.read(reinterpret_cast<char*>(state_data), sizeof(state_data));
        ifs.read(reinterpret_cast<char*>(&keys_processed), sizeof(keys_processed));

        if (state_data[0] == wu.unit_id) {
            wu.state = static_cast<WorkUnitState>(state_data[1]);
            wu.reassign_count = state_data[2];
            wu.keys_processed = keys_processed;

            // Reset assigned state for resume
            if (wu.state == WorkUnitState::ASSIGNED) {
                wu.state = WorkUnitState::AVAILABLE;
                wu.assigned_client_id = 0;
            }
        }
    }

    // Check if found
    if (header.found) {
        found_.store(true);
        arith256::copy(header.found_scalar, found_scalar_);
    }

    std::cout << "[Server] Loaded checkpoint: " << header.completed_units << "/"
              << header.total_units << " units completed\n";

    return true;
}

// ============================================================================
// STATUS
// ============================================================================

void CycloneServer::print_status() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();

    uint32_t available = work_manager_.count_by_state(WorkUnitState::AVAILABLE);
    uint32_t assigned = work_manager_.count_by_state(WorkUnitState::ASSIGNED);
    uint32_t completed = work_manager_.count_by_state(WorkUnitState::COMPLETED);
    uint32_t total = work_manager_.get_total_units();

    double progress = (total > 0) ? (100.0 * completed / total) : 0.0;
    double speed = get_aggregate_speed();
    uint64_t keys = get_total_keys_processed();

    std::cout << "\r[" << format_time(elapsed) << "] "
              << "Units: " << completed << "/" << total << " (" << std::fixed << std::setprecision(1) << progress << "%) "
              << "| Active: " << assigned << " | Pending: " << available << " "
              << "| Speed: " << format_speed(speed) << " "
              << "| Keys: " << format_keys(keys);

    // Count connected clients
    uint32_t connected = 0;
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(clients_mutex_));
        for (const auto& [id, client] : clients_) {
            if (client.connected) ++connected;
        }
    }
    std::cout << " | Clients: " << connected;

    std::cout << "    " << std::flush;
}

uint64_t CycloneServer::get_total_keys_processed() const {
    return work_manager_.get_total_keys_processed();
}

double CycloneServer::get_aggregate_speed() const {
    double total = 0.0;
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(clients_mutex_));
    for (const auto& [id, client] : clients_) {
        if (client.connected) {
            total += client.current_speed_gkeys;
        }
    }
    return total;
}

// ============================================================================
// UTILITIES
// ============================================================================

uint64_t CycloneServer::current_time_sec() const {
    return static_cast<uint64_t>(std::time(nullptr));
}

std::string CycloneServer::format_time(uint64_t seconds) const {
    uint64_t h = seconds / 3600;
    uint64_t m = (seconds % 3600) / 60;
    uint64_t s = seconds % 60;

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << h << ":"
        << std::setw(2) << m << ":" << std::setw(2) << s;
    return oss.str();
}

std::string CycloneServer::format_speed(double gkeys) const {
    std::ostringstream oss;
    if (gkeys >= 1.0) {
        oss << std::fixed << std::setprecision(2) << gkeys << " Gkeys/s";
    } else if (gkeys >= 0.001) {
        oss << std::fixed << std::setprecision(2) << (gkeys * 1000.0) << " Mkeys/s";
    } else {
        oss << std::fixed << std::setprecision(2) << (gkeys * 1000000.0) << " Kkeys/s";
    }
    return oss.str();
}

std::string CycloneServer::format_keys(uint64_t keys) const {
    std::ostringstream oss;
    if (keys >= 1e15) {
        oss << std::fixed << std::setprecision(2) << (keys / 1e15) << "P";
    } else if (keys >= 1e12) {
        oss << std::fixed << std::setprecision(2) << (keys / 1e12) << "T";
    } else if (keys >= 1e9) {
        oss << std::fixed << std::setprecision(2) << (keys / 1e9) << "G";
    } else if (keys >= 1e6) {
        oss << std::fixed << std::setprecision(2) << (keys / 1e6) << "M";
    } else {
        oss << keys;
    }
    return oss.str();
}

// ============================================================================
// COMMAND-LINE PARSING
// ============================================================================

void print_server_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  --range <start>:<end>       Search range (hex, power-of-2 aligned)\n";
    std::cout << "  --address <P2PKH>           Target Bitcoin address\n";
    std::cout << "  OR --target-hash160 <hex>   Target hash160 directly\n\n";
    std::cout << "Options:\n";
    std::cout << "  --port <N>                  Server port (default: 17403)\n";
    std::cout << "  --unit-bits <N>             Work unit size as 2^N keys (default: 36)\n";
    std::cout << "  --batch-size <N>            Recommended batch size (default: 128)\n";
    std::cout << "  --slices <N>                Recommended slices (default: 16)\n";
    std::cout << "  --pincer                    Enable bidirectional mode\n";
    std::cout << "  --kxe                       Enable KXE permuted scanning mode\n";
    std::cout << "  --kxe-seed <N>              KXE permutation seed (default: random)\n";
    std::cout << "  --checkpoint <file>         Checkpoint file path\n";
    std::cout << "  --checkpoint-interval <N>   Checkpoint interval in seconds (default: 300)\n";
    std::cout << "  --heartbeat-timeout <N>     Client timeout in seconds (default: 90)\n";
    std::cout << "  --max-clients <N>           Maximum clients (default: 256)\n";
    std::cout << "  -h, --help                  Show this help\n";
}

bool parse_server_args(int argc, char* argv[], ServerConfig& config) {
    bool has_range = false;
    bool has_target = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_server_usage(argv[0]);
            exit(0);
        }
        else if (arg == "--range" && i + 1 < argc) {
            std::string range = argv[++i];
            size_t colon = range.find(':');
            if (colon == std::string::npos) {
                std::cerr << "Invalid range format. Use start:end\n";
                return false;
            }
            if (!arith256::from_hex(range.substr(0, colon), config.range_start) ||
                !arith256::from_hex(range.substr(colon + 1), config.range_end)) {
                std::cerr << "Invalid range hex values\n";
                return false;
            }
            has_range = true;
        }
        else if (arg == "--address" && i + 1 < argc) {
            // TODO: Implement Base58 decoding for P2PKH addresses
            std::cerr << "P2PKH address decoding not yet implemented. Use --target-hash160\n";
            return false;
        }
        else if (arg == "--target-hash160" && i + 1 < argc) {
            std::string hex = argv[++i];
            if (hex.size() != 40) {
                std::cerr << "Invalid hash160 length\n";
                return false;
            }
            for (int j = 0; j < 20; ++j) {
                config.target_hash160[j] = static_cast<uint8_t>(
                    std::stoul(hex.substr(j * 2, 2), nullptr, 16));
            }
            has_target = true;
        }
        else if (arg == "--port" && i + 1 < argc) {
            config.port = static_cast<uint16_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--unit-bits" && i + 1 < argc) {
            config.work_unit_bits = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--slices" && i + 1 < argc) {
            config.slices_per_launch = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--pincer") {
            config.pincer_mode = true;
        }
        else if (arg == "--kxe") {
            config.kxe_mode = true;
        }
        else if (arg == "--kxe-seed" && i + 1 < argc) {
            config.kxe_seed = std::stoull(argv[++i]);
        }
        else if (arg == "--checkpoint" && i + 1 < argc) {
            config.checkpoint_file = argv[++i];
        }
        else if (arg == "--checkpoint-interval" && i + 1 < argc) {
            config.checkpoint_interval_sec = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--heartbeat-timeout" && i + 1 < argc) {
            config.heartbeat_timeout_sec = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--max-clients" && i + 1 < argc) {
            config.max_clients = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (!has_range) {
        std::cerr << "Missing required --range argument\n";
        return false;
    }
    if (!has_target) {
        std::cerr << "Missing required --address or --target-hash160 argument\n";
        return false;
    }

    // Validate range
    if (!validate_range(config.range_start, config.range_end)) {
        std::cerr << "Invalid range: must be power-of-2 length and properly aligned\n";
        return false;
    }

    return true;
}

// ============================================================================
// MAIN (standalone server)
// ============================================================================

int main(int argc, char* argv[]) {
    ServerConfig config;

    if (!parse_server_args(argc, argv, config)) {
        print_server_usage(argv[0]);
        return 1;
    }

    CycloneServer server(config);

    if (!server.start()) {
        return 1;
    }

    server.wait_for_completion();

    return 0;
}

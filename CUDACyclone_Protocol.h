// ============================================================================
// CUDACyclone Distributed Mode - Network Protocol Definitions
// ============================================================================
// Binary protocol for server/client communication
// Default port: 17403
// All multi-byte values are little-endian
// ============================================================================

#ifndef CUDACYCLONE_PROTOCOL_H
#define CUDACYCLONE_PROTOCOL_H

#include <cstdint>
#include <cstring>

// Protocol constants
constexpr uint16_t DEFAULT_SERVER_PORT = 17403;
constexpr uint32_t PROTOCOL_VERSION = 2;  // v2 adds KXE mode support
constexpr uint32_t PROTOCOL_MAGIC = 0x4343594B;  // "CCYK" in ASCII

// Timeouts and limits
constexpr uint32_t DEFAULT_HEARTBEAT_INTERVAL_SEC = 30;
constexpr uint32_t DEFAULT_HEARTBEAT_TIMEOUT_SEC = 90;
constexpr uint32_t DEFAULT_PROGRESS_INTERVAL_SEC = 10;
constexpr uint32_t MAX_CLIENTS = 256;
constexpr uint32_t MAX_GPUS_PER_CLIENT = 16;
constexpr uint32_t MAX_WORK_UNITS_PER_REQUEST = 4;
constexpr uint32_t MAX_REASSIGN_COUNT = 3;

// Work unit size constraints (must be power of 2)
constexpr uint32_t MIN_WORK_UNIT_BITS = 28;  // 2^28 = 268M keys minimum
constexpr uint32_t MAX_WORK_UNIT_BITS = 48;  // 2^48 = 281T keys maximum
constexpr uint32_t DEFAULT_WORK_UNIT_BITS = 36;  // 2^36 = 68.7B keys

// ============================================================================
// MESSAGE TYPES
// ============================================================================

enum class MessageType : uint16_t {
    // Client -> Server (0x00xx)
    REGISTER_REQUEST    = 0x0001,
    HEARTBEAT           = 0x0002,
    WORK_REQUEST        = 0x0003,
    PROGRESS_REPORT     = 0x0004,
    UNIT_COMPLETE       = 0x0005,
    FOUND_RESULT        = 0x0006,
    DISCONNECT          = 0x0007,

    // Server -> Client (0x01xx)
    REGISTER_RESPONSE   = 0x0101,
    HEARTBEAT_ACK       = 0x0102,
    WORK_ASSIGNMENT     = 0x0103,
    NO_WORK_AVAILABLE   = 0x0104,
    SEARCH_COMPLETE     = 0x0105,
    KEY_FOUND           = 0x0106,
    SERVER_SHUTDOWN     = 0x0107,
    ERROR_RESPONSE      = 0x01FF
};

// ============================================================================
// ERROR CODES
// ============================================================================

enum class ErrorCode : uint32_t {
    OK                      = 0,
    VERSION_MISMATCH        = 1,
    INVALID_MESSAGE         = 2,
    CLIENT_NOT_REGISTERED   = 3,
    INVALID_WORK_UNIT       = 4,
    DUPLICATE_CLIENT        = 5,
    SERVER_FULL             = 6,
    INVALID_RESULT          = 7,
    INTERNAL_ERROR          = 100
};

// ============================================================================
// WORK UNIT STATE
// ============================================================================

enum class WorkUnitState : uint8_t {
    AVAILABLE   = 0,   // Ready to be assigned
    ASSIGNED    = 1,   // Currently being processed by a client
    COMPLETED   = 2,   // Successfully finished
    EXPIRED     = 3,   // Client timed out, pending reassignment
    VERIFIED    = 4    // Completed and result verified
};

// ============================================================================
// SCAN DIRECTION (for bidirectional/pincer mode)
// ============================================================================

enum class ScanDirection : uint8_t {
    FORWARD     = 0,   // Scan from start toward end (add B each batch)
    BACKWARD    = 1    // Scan from end toward start (subtract B each batch)
};

// ============================================================================
// SCAN MODE (sequential vs KXE permuted)
// ============================================================================

enum class ScanMode : uint8_t {
    SEQUENTIAL  = 0,   // Traditional sequential scanning
    KXE         = 1    // KXE permuted block scanning
};

// ============================================================================
// MESSAGE STRUCTURES
// ============================================================================
// All structures are packed for network transmission

#pragma pack(push, 1)

// ----------------------------------------------------------------------------
// Message Header (8 bytes) - prefixes ALL messages
// ----------------------------------------------------------------------------
struct MessageHeader {
    uint32_t magic;          // PROTOCOL_MAGIC
    uint16_t msg_type;       // MessageType enum
    uint16_t payload_size;   // Size of payload after this header

    MessageHeader() : magic(PROTOCOL_MAGIC), msg_type(0), payload_size(0) {}

    MessageHeader(MessageType type, uint16_t size)
        : magic(PROTOCOL_MAGIC), msg_type(static_cast<uint16_t>(type)), payload_size(size) {}

    bool is_valid() const { return magic == PROTOCOL_MAGIC; }
    MessageType type() const { return static_cast<MessageType>(msg_type); }
};

// ----------------------------------------------------------------------------
// GPU Information (80 bytes)
// ----------------------------------------------------------------------------
struct GPUInfoMsg {
    char name[64];                  // GPU name (null-terminated)
    uint32_t compute_cap_major;     // Compute capability major
    uint32_t compute_cap_minor;     // Compute capability minor
    uint64_t memory_bytes;          // Total GPU memory

    GPUInfoMsg() {
        memset(name, 0, sizeof(name));
        compute_cap_major = 0;
        compute_cap_minor = 0;
        memory_bytes = 0;
    }
};

// ----------------------------------------------------------------------------
// CLIENT -> SERVER MESSAGES
// ----------------------------------------------------------------------------

// REGISTER_REQUEST (variable size: 20 + hostname_len + gpu_count * 80)
struct RegisterRequestMsg {
    uint32_t protocol_version;      // Must match PROTOCOL_VERSION
    uint32_t gpu_count;             // Number of GPUs on this client
    uint32_t batch_size;            // Preferred batch size
    uint32_t slices;                // Preferred slices per launch
    uint8_t  supports_pincer;       // 1 if client supports bidirectional mode
    uint8_t  hostname_len;          // Length of hostname string
    uint8_t  reserved[2];           // Padding
    // Followed by:
    // - char hostname[hostname_len]
    // - GPUInfoMsg gpu_info[gpu_count]

    RegisterRequestMsg() {
        protocol_version = PROTOCOL_VERSION;
        gpu_count = 0;
        batch_size = 128;
        slices = 16;
        supports_pincer = 1;
        hostname_len = 0;
        reserved[0] = reserved[1] = 0;
    }
};

// HEARTBEAT (16 bytes)
struct HeartbeatMsg {
    uint32_t client_id;             // Assigned client ID
    uint32_t active_unit_count;     // Number of active work units
    double   current_speed_gkeys;   // Current aggregate speed (Gkeys/s)

    HeartbeatMsg() : client_id(0), active_unit_count(0), current_speed_gkeys(0.0) {}
};

// WORK_REQUEST (8 bytes)
struct WorkRequestMsg {
    uint32_t client_id;             // Client requesting work
    uint32_t units_requested;       // How many units client wants (1-4)

    WorkRequestMsg() : client_id(0), units_requested(1) {}
};

// PROGRESS_REPORT (32 bytes)
struct ProgressReportMsg {
    uint32_t client_id;             // Client ID
    uint32_t unit_id;               // Work unit ID
    uint64_t keys_processed;        // Keys processed in this unit so far
    double   speed_gkeys;           // Current speed on this unit
    uint8_t  status;                // 0=working, 1=completed, 2=error
    uint8_t  reserved[7];           // Padding

    ProgressReportMsg() {
        client_id = 0;
        unit_id = 0;
        keys_processed = 0;
        speed_gkeys = 0.0;
        status = 0;
        memset(reserved, 0, sizeof(reserved));
    }
};

// UNIT_COMPLETE (24 bytes)
struct UnitCompleteMsg {
    uint32_t client_id;             // Client ID
    uint32_t unit_id;               // Work unit ID
    uint64_t keys_processed;        // Total keys processed in unit
    double   avg_speed_gkeys;       // Average speed during processing

    UnitCompleteMsg() : client_id(0), unit_id(0), keys_processed(0), avg_speed_gkeys(0.0) {}
};

// FOUND_RESULT (128 bytes)
struct FoundResultMsg {
    uint32_t client_id;             // Client that found it
    uint32_t unit_id;               // Work unit where found
    uint64_t scalar[4];             // 256-bit private key (little-endian limbs)
    uint8_t  hash160[20];           // Hash160 for verification
    uint64_t pubkey_x[4];           // Public key X coordinate
    uint64_t pubkey_y[4];           // Public key Y coordinate
    uint32_t reserved;              // Padding

    FoundResultMsg() {
        client_id = 0;
        unit_id = 0;
        memset(scalar, 0, sizeof(scalar));
        memset(hash160, 0, sizeof(hash160));
        memset(pubkey_x, 0, sizeof(pubkey_x));
        memset(pubkey_y, 0, sizeof(pubkey_y));
        reserved = 0;
    }
};

// DISCONNECT (4 bytes)
struct DisconnectMsg {
    uint32_t client_id;             // Client disconnecting

    DisconnectMsg() : client_id(0) {}
};

// ----------------------------------------------------------------------------
// SERVER -> CLIENT MESSAGES
// ----------------------------------------------------------------------------

// REGISTER_RESPONSE (40 bytes)
struct RegisterResponseMsg {
    uint32_t status;                // ErrorCode (0 = OK)
    uint32_t client_id;             // Assigned client ID (0 if rejected)
    uint32_t heartbeat_interval;    // Required heartbeat interval (seconds)
    uint32_t progress_interval;     // Required progress report interval (seconds)
    uint32_t max_units_per_request; // Max work units per request
    uint8_t  scan_mode;             // ScanMode enum (0=sequential, 1=KXE)
    uint8_t  reserved1[3];          // Padding
    uint64_t kxe_seed;              // KXE permutation seed (if KXE mode)
    uint64_t total_blocks;          // Total blocks in search (for KXE progress)

    RegisterResponseMsg() {
        status = static_cast<uint32_t>(ErrorCode::OK);
        client_id = 0;
        heartbeat_interval = DEFAULT_HEARTBEAT_INTERVAL_SEC;
        progress_interval = DEFAULT_PROGRESS_INTERVAL_SEC;
        max_units_per_request = MAX_WORK_UNITS_PER_REQUEST;
        scan_mode = static_cast<uint8_t>(ScanMode::SEQUENTIAL);
        reserved1[0] = reserved1[1] = reserved1[2] = 0;
        kxe_seed = 0;
        total_blocks = 0;
    }
};

// HEARTBEAT_ACK (8 bytes)
struct HeartbeatAckMsg {
    uint32_t status;                // ErrorCode (0 = OK)
    uint32_t server_time;           // Server timestamp (for sync)

    HeartbeatAckMsg() : status(0), server_time(0) {}
};

// WORK_ASSIGNMENT (128 bytes per unit)
struct WorkAssignmentMsg {
    uint32_t unit_id;               // Work unit ID
    uint64_t range_start[4];        // 256-bit range start (global, for KXE base)
    uint64_t range_end[4];          // 256-bit range end (global)
    uint8_t  target_hash160[20];    // Target address hash
    uint32_t batch_size;            // Batch size to use
    uint32_t slices;                // Slices per launch
    uint8_t  scan_direction;        // ScanDirection enum
    uint8_t  pincer_enabled;        // 1 if this is part of pincer pair
    uint16_t pincer_partner_unit;   // Partner unit ID (if pincer)
    uint8_t  scan_mode;             // ScanMode enum (0=sequential, 1=KXE)
    uint8_t  reserved1[3];          // Padding
    uint64_t kxe_block_index;       // Block index to process (KXE mode)
    uint64_t kxe_seed;              // Permutation seed (KXE mode)
    uint64_t keys_per_block;        // Keys per block (KXE mode)

    WorkAssignmentMsg() {
        unit_id = 0;
        memset(range_start, 0, sizeof(range_start));
        memset(range_end, 0, sizeof(range_end));
        memset(target_hash160, 0, sizeof(target_hash160));
        batch_size = 128;
        slices = 16;
        scan_direction = static_cast<uint8_t>(ScanDirection::FORWARD);
        pincer_enabled = 0;
        pincer_partner_unit = 0;
        scan_mode = static_cast<uint8_t>(ScanMode::SEQUENTIAL);
        reserved1[0] = reserved1[1] = reserved1[2] = 0;
        kxe_block_index = 0;
        kxe_seed = 0;
        keys_per_block = 0;
    }
};

// NO_WORK_AVAILABLE (8 bytes)
struct NoWorkAvailableMsg {
    uint32_t reason;                // 0=all assigned, 1=search complete, 2=paused
    uint32_t retry_after_sec;       // Suggested retry delay

    NoWorkAvailableMsg() : reason(0), retry_after_sec(5) {}
};

// SEARCH_COMPLETE (16 bytes) - sent when all work units are done
struct SearchCompleteMsg {
    uint64_t total_keys_searched;   // Total keys searched across all clients
    uint32_t total_work_units;      // Total work units processed
    uint32_t found;                 // 1 if key was found, 0 otherwise

    SearchCompleteMsg() : total_keys_searched(0), total_work_units(0), found(0) {}
};

// KEY_FOUND (84 bytes) - broadcast to all clients when key is found
struct KeyFoundMsg {
    uint32_t finder_client_id;      // Client that found it
    uint32_t unit_id;               // Work unit where found
    uint64_t scalar[4];             // 256-bit private key
    char     address[36];           // Base58 P2PKH address (null-terminated)
    uint64_t search_time_sec;       // Total search time

    KeyFoundMsg() {
        finder_client_id = 0;
        unit_id = 0;
        memset(scalar, 0, sizeof(scalar));
        memset(address, 0, sizeof(address));
        search_time_sec = 0;
    }
};

// SERVER_SHUTDOWN (8 bytes)
struct ServerShutdownMsg {
    uint32_t reason;                // 0=normal, 1=error, 2=maintenance
    uint32_t reserved;

    ServerShutdownMsg() : reason(0), reserved(0) {}
};

// ERROR_RESPONSE (variable size: 8 + message_len)
struct ErrorResponseMsg {
    uint32_t error_code;            // ErrorCode enum
    uint16_t message_len;           // Length of error message
    uint16_t reserved;
    // Followed by: char message[message_len]

    ErrorResponseMsg() : error_code(0), message_len(0), reserved(0) {}
};

#pragma pack(pop)

// ============================================================================
// STRUCT SIZE VERIFICATION
// ============================================================================
// Verify packed struct sizes at compile time to prevent protocol mismatches

static_assert(sizeof(MessageHeader) == 8, "MessageHeader size mismatch");
static_assert(sizeof(GPUInfoMsg) == 80, "GPUInfoMsg size mismatch");
static_assert(sizeof(RegisterRequestMsg) == 20, "RegisterRequestMsg size mismatch");
static_assert(sizeof(HeartbeatMsg) == 16, "HeartbeatMsg size mismatch");
static_assert(sizeof(WorkRequestMsg) == 8, "WorkRequestMsg size mismatch");
static_assert(sizeof(ProgressReportMsg) == 32, "ProgressReportMsg size mismatch");
static_assert(sizeof(UnitCompleteMsg) == 24, "UnitCompleteMsg size mismatch");
static_assert(sizeof(FoundResultMsg) == 128, "FoundResultMsg size mismatch");
static_assert(sizeof(DisconnectMsg) == 4, "DisconnectMsg size mismatch");
static_assert(sizeof(RegisterResponseMsg) == 40, "RegisterResponseMsg size mismatch");
static_assert(sizeof(HeartbeatAckMsg) == 8, "HeartbeatAckMsg size mismatch");
static_assert(sizeof(WorkAssignmentMsg) == 128, "WorkAssignmentMsg size mismatch");
static_assert(sizeof(NoWorkAvailableMsg) == 8, "NoWorkAvailableMsg size mismatch");
static_assert(sizeof(SearchCompleteMsg) == 16, "SearchCompleteMsg size mismatch");
static_assert(sizeof(KeyFoundMsg) == 84, "KeyFoundMsg size mismatch");
static_assert(sizeof(ServerShutdownMsg) == 8, "ServerShutdownMsg size mismatch");
static_assert(sizeof(ErrorResponseMsg) == 8, "ErrorResponseMsg size mismatch");

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Calculate total message size for variable-length messages
inline size_t calc_register_request_size(uint8_t hostname_len, uint32_t gpu_count) {
    return sizeof(MessageHeader) + sizeof(RegisterRequestMsg) + hostname_len + (gpu_count * sizeof(GPUInfoMsg));
}

inline size_t calc_error_response_size(uint16_t message_len) {
    return sizeof(MessageHeader) + sizeof(ErrorResponseMsg) + message_len;
}

// Validate message header
inline bool validate_header(const MessageHeader& hdr, size_t received_size) {
    if (!hdr.is_valid()) return false;
    if (received_size < sizeof(MessageHeader)) return false;
    if (received_size < sizeof(MessageHeader) + hdr.payload_size) return false;
    return true;
}

// Get message type name for logging
inline const char* message_type_name(MessageType type) {
    switch (type) {
        case MessageType::REGISTER_REQUEST:     return "REGISTER_REQUEST";
        case MessageType::HEARTBEAT:            return "HEARTBEAT";
        case MessageType::WORK_REQUEST:         return "WORK_REQUEST";
        case MessageType::PROGRESS_REPORT:      return "PROGRESS_REPORT";
        case MessageType::UNIT_COMPLETE:        return "UNIT_COMPLETE";
        case MessageType::FOUND_RESULT:         return "FOUND_RESULT";
        case MessageType::DISCONNECT:           return "DISCONNECT";
        case MessageType::REGISTER_RESPONSE:    return "REGISTER_RESPONSE";
        case MessageType::HEARTBEAT_ACK:        return "HEARTBEAT_ACK";
        case MessageType::WORK_ASSIGNMENT:      return "WORK_ASSIGNMENT";
        case MessageType::NO_WORK_AVAILABLE:    return "NO_WORK_AVAILABLE";
        case MessageType::SEARCH_COMPLETE:      return "SEARCH_COMPLETE";
        case MessageType::KEY_FOUND:            return "KEY_FOUND";
        case MessageType::SERVER_SHUTDOWN:      return "SERVER_SHUTDOWN";
        case MessageType::ERROR_RESPONSE:       return "ERROR_RESPONSE";
        default:                                return "UNKNOWN";
    }
}

// Get error code name for logging
inline const char* error_code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::OK:                     return "OK";
        case ErrorCode::VERSION_MISMATCH:       return "VERSION_MISMATCH";
        case ErrorCode::INVALID_MESSAGE:        return "INVALID_MESSAGE";
        case ErrorCode::CLIENT_NOT_REGISTERED:  return "CLIENT_NOT_REGISTERED";
        case ErrorCode::INVALID_WORK_UNIT:      return "INVALID_WORK_UNIT";
        case ErrorCode::DUPLICATE_CLIENT:       return "DUPLICATE_CLIENT";
        case ErrorCode::SERVER_FULL:            return "SERVER_FULL";
        case ErrorCode::INVALID_RESULT:         return "INVALID_RESULT";
        case ErrorCode::INTERNAL_ERROR:         return "INTERNAL_ERROR";
        default:                                return "UNKNOWN";
    }
}

#endif // CUDACYCLONE_PROTOCOL_H

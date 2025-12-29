// ============================================================================
// CUDACyclone Distributed Mode - Network Utilities
// ============================================================================
// Cross-platform socket utilities for TCP communication
// ============================================================================

#ifndef CUDACYCLONE_NETWORK_H
#define CUDACYCLONE_NETWORK_H

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <memory>

#include "CUDACyclone_Protocol.h"

// Platform-specific includes
#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define INVALID_SOCKET_VALUE INVALID_SOCKET
    #define SOCKET_ERROR_VALUE SOCKET_ERROR
    #define CLOSE_SOCKET(s) closesocket(s)
    #define GET_SOCKET_ERROR() WSAGetLastError()
#else
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
    #include <poll.h>
    typedef int socket_t;
    #define INVALID_SOCKET_VALUE (-1)
    #define SOCKET_ERROR_VALUE (-1)
    #define CLOSE_SOCKET(s) close(s)
    #define GET_SOCKET_ERROR() errno
#endif

namespace net {

// ============================================================================
// INITIALIZATION
// ============================================================================

// Initialize network subsystem (required on Windows)
bool initialize();

// Cleanup network subsystem
void cleanup();

// ============================================================================
// SOCKET UTILITIES
// ============================================================================

// Set socket to non-blocking mode
bool set_nonblocking(socket_t sock, bool nonblocking);

// Set TCP_NODELAY (disable Nagle's algorithm)
bool set_nodelay(socket_t sock, bool nodelay);

// Set SO_REUSEADDR
bool set_reuseaddr(socket_t sock, bool reuse);

// Set socket receive/send timeout (milliseconds)
bool set_recv_timeout(socket_t sock, uint32_t timeout_ms);
bool set_send_timeout(socket_t sock, uint32_t timeout_ms);

// Set SO_KEEPALIVE with optional parameters
bool set_keepalive(socket_t sock, bool enable, int idle_sec = 60, int interval_sec = 10, int count = 5);

// Get last socket error as string
std::string get_last_error_string();

// ============================================================================
// CONNECTION MANAGEMENT
// ============================================================================

// Create a TCP server socket bound to the given port
// Returns INVALID_SOCKET_VALUE on failure
socket_t create_server_socket(uint16_t port, int backlog = 10);

// Accept a connection on a server socket
// Returns INVALID_SOCKET_VALUE on failure
// If timeout_ms > 0, waits up to that time for a connection
socket_t accept_connection(socket_t server_sock, std::string& client_addr, uint16_t& client_port, int timeout_ms = -1);

// Connect to a remote server
// Returns INVALID_SOCKET_VALUE on failure
socket_t connect_to_server(const std::string& host, uint16_t port, int timeout_ms = 5000);

// Close a socket gracefully
void close_socket(socket_t sock);

// ============================================================================
// DATA TRANSMISSION
// ============================================================================

// Send exactly 'size' bytes
// Returns true if all bytes sent, false on error
bool send_all(socket_t sock, const void* data, size_t size);

// Receive exactly 'size' bytes
// Returns true if all bytes received, false on error/disconnect
bool recv_all(socket_t sock, void* buffer, size_t size);

// Receive with timeout (milliseconds)
// Returns number of bytes received, 0 on timeout, -1 on error
ssize_t recv_with_timeout(socket_t sock, void* buffer, size_t size, int timeout_ms);

// Check if socket has data available to read
// Returns 1 if data available, 0 if timeout, -1 on error
int poll_readable(socket_t sock, int timeout_ms);

// Check if socket is writable
int poll_writable(socket_t sock, int timeout_ms);

// ============================================================================
// MESSAGE TRANSMISSION
// ============================================================================

// Send a complete message (header + payload)
bool send_message(socket_t sock, MessageType type, const void* payload, uint16_t payload_size);

// Send a message with no payload
bool send_message(socket_t sock, MessageType type);

// Receive a message header
// Returns false on error/disconnect
bool recv_message_header(socket_t sock, MessageHeader& header, int timeout_ms = -1);

// Receive message payload after header
bool recv_message_payload(socket_t sock, void* buffer, size_t size);

// ============================================================================
// UTILITY CLASSES
// ============================================================================

// RAII wrapper for socket
class Socket {
public:
    Socket() : sock_(INVALID_SOCKET_VALUE) {}
    explicit Socket(socket_t sock) : sock_(sock) {}
    ~Socket() { close(); }

    // Non-copyable
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    // Movable
    Socket(Socket&& other) noexcept : sock_(other.sock_) { other.sock_ = INVALID_SOCKET_VALUE; }
    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            close();
            sock_ = other.sock_;
            other.sock_ = INVALID_SOCKET_VALUE;
        }
        return *this;
    }

    void close() {
        if (sock_ != INVALID_SOCKET_VALUE) {
            close_socket(sock_);
            sock_ = INVALID_SOCKET_VALUE;
        }
    }

    socket_t get() const { return sock_; }
    socket_t release() { socket_t s = sock_; sock_ = INVALID_SOCKET_VALUE; return s; }
    bool valid() const { return sock_ != INVALID_SOCKET_VALUE; }
    operator bool() const { return valid(); }

    // Convenience wrappers
    bool send(const void* data, size_t size) { return send_all(sock_, data, size); }
    bool recv(void* buffer, size_t size) { return recv_all(sock_, buffer, size); }
    bool send_msg(MessageType type, const void* payload = nullptr, uint16_t size = 0) {
        return send_message(sock_, type, payload, size);
    }

private:
    socket_t sock_;
};

// Thread-safe socket wrapper with mutex protection
class ThreadSafeSocket {
public:
    ThreadSafeSocket() : sock_(INVALID_SOCKET_VALUE) {}
    explicit ThreadSafeSocket(socket_t sock) : sock_(sock) {}

    void set(socket_t sock) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sock_ != INVALID_SOCKET_VALUE) {
            close_socket(sock_);
        }
        sock_ = sock;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sock_ != INVALID_SOCKET_VALUE) {
            close_socket(sock_);
            sock_ = INVALID_SOCKET_VALUE;
        }
    }

    bool send(const void* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sock_ == INVALID_SOCKET_VALUE) return false;
        return send_all(sock_, data, size);
    }

    bool recv(void* buffer, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sock_ == INVALID_SOCKET_VALUE) return false;
        return recv_all(sock_, buffer, size);
    }

    bool send_message(MessageType type, const void* payload = nullptr, uint16_t size = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sock_ == INVALID_SOCKET_VALUE) return false;
        return net::send_message(sock_, type, payload, size);
    }

    bool valid() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sock_ != INVALID_SOCKET_VALUE;
    }

    socket_t get_unsafe() const { return sock_; }  // Use with caution

private:
    socket_t sock_;
    mutable std::mutex mutex_;
};

// ============================================================================
// ADDRESS UTILITIES
// ============================================================================

// Parse "host:port" string
bool parse_address(const std::string& addr, std::string& host, uint16_t& port);

// Format address to string
std::string format_address(const std::string& host, uint16_t port);

// Resolve hostname to IP address
bool resolve_hostname(const std::string& hostname, std::string& ip_address);

// Get local hostname
std::string get_hostname();

// Get local IP address (first non-loopback interface)
std::string get_local_ip();

} // namespace net

#endif // CUDACYCLONE_NETWORK_H

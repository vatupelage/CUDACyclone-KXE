// ============================================================================
// CUDACyclone Distributed Mode - Network Utilities Implementation
// ============================================================================

#include "CUDACyclone_Network.h"
#include <cstring>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
    #include <iphlpapi.h>
    #pragma comment(lib, "iphlpapi.lib")
#else
    #include <ifaddrs.h>
    #include <net/if.h>
#endif

namespace net {

// ============================================================================
// INITIALIZATION
// ============================================================================

static std::atomic<bool> g_initialized{false};

bool initialize() {
#ifdef _WIN32
    if (g_initialized.exchange(true)) return true;

    WSADATA wsa_data;
    int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (result != 0) {
        g_initialized = false;
        return false;
    }
    return true;
#else
    g_initialized = true;
    return true;
#endif
}

void cleanup() {
#ifdef _WIN32
    if (g_initialized.exchange(false)) {
        WSACleanup();
    }
#else
    g_initialized = false;
#endif
}

// ============================================================================
// SOCKET UTILITIES
// ============================================================================

bool set_nonblocking(socket_t sock, bool nonblocking) {
#ifdef _WIN32
    u_long mode = nonblocking ? 1 : 0;
    return ioctlsocket(sock, FIONBIO, &mode) == 0;
#else
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1) return false;
    if (nonblocking) {
        flags |= O_NONBLOCK;
    } else {
        flags &= ~O_NONBLOCK;
    }
    return fcntl(sock, F_SETFL, flags) != -1;
#endif
}

bool set_nodelay(socket_t sock, bool nodelay) {
    int flag = nodelay ? 1 : 0;
    return setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag)) == 0;
}

bool set_reuseaddr(socket_t sock, bool reuse) {
    int flag = reuse ? 1 : 0;
    return setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(flag)) == 0;
}

bool set_recv_timeout(socket_t sock, uint32_t timeout_ms) {
#ifdef _WIN32
    DWORD tv = timeout_ms;
    return setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv)) == 0;
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    return setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
#endif
}

bool set_send_timeout(socket_t sock, uint32_t timeout_ms) {
#ifdef _WIN32
    DWORD tv = timeout_ms;
    return setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv)) == 0;
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    return setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
#endif
}

bool set_keepalive(socket_t sock, bool enable, int idle_sec, int interval_sec, int count) {
    int flag = enable ? 1 : 0;
    if (setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (const char*)&flag, sizeof(flag)) != 0) {
        return false;
    }

    if (!enable) return true;

#ifdef _WIN32
    // Windows doesn't support fine-grained keepalive settings easily
    // Would need WSAIoctl with SIO_KEEPALIVE_VALS
    (void)idle_sec;
    (void)interval_sec;
    (void)count;
    return true;
#else
    #ifdef TCP_KEEPIDLE
    if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, &idle_sec, sizeof(idle_sec)) != 0) {
        return false;
    }
    #endif
    #ifdef TCP_KEEPINTVL
    if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &interval_sec, sizeof(interval_sec)) != 0) {
        return false;
    }
    #endif
    #ifdef TCP_KEEPCNT
    if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count)) != 0) {
        return false;
    }
    #endif
    return true;
#endif
}

std::string get_last_error_string() {
#ifdef _WIN32
    int err = WSAGetLastError();
    char* msg = nullptr;
    FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&msg, 0, nullptr);
    std::string result = msg ? msg : "Unknown error";
    if (msg) LocalFree(msg);
    return result;
#else
    return strerror(errno);
#endif
}

// ============================================================================
// CONNECTION MANAGEMENT
// ============================================================================

socket_t create_server_socket(uint16_t port, int backlog) {
    socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET_VALUE) {
        return INVALID_SOCKET_VALUE;
    }

    // Set socket options
    set_reuseaddr(sock, true);
    set_nodelay(sock, true);

    // Bind to port
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close_socket(sock);
        return INVALID_SOCKET_VALUE;
    }

    // Start listening
    if (listen(sock, backlog) != 0) {
        close_socket(sock);
        return INVALID_SOCKET_VALUE;
    }

    return sock;
}

socket_t accept_connection(socket_t server_sock, std::string& client_addr, uint16_t& client_port, int timeout_ms) {
    // Wait for connection if timeout specified
    if (timeout_ms > 0) {
        int result = poll_readable(server_sock, timeout_ms);
        if (result <= 0) {
            return INVALID_SOCKET_VALUE;
        }
    }

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    socket_t client_sock = accept(server_sock, (struct sockaddr*)&addr, &addr_len);

    if (client_sock == INVALID_SOCKET_VALUE) {
        return INVALID_SOCKET_VALUE;
    }

    // Extract client address
    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str));
    client_addr = ip_str;
    client_port = ntohs(addr.sin_port);

    // Configure client socket
    set_nodelay(client_sock, true);
    set_keepalive(client_sock, true);

    return client_sock;
}

socket_t connect_to_server(const std::string& host, uint16_t port, int timeout_ms) {
    // Resolve hostname
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    std::string port_str = std::to_string(port);
    int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (ret != 0) {
        return INVALID_SOCKET_VALUE;
    }

    socket_t sock = INVALID_SOCKET_VALUE;

    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == INVALID_SOCKET_VALUE) {
            continue;
        }

        // Set non-blocking for timeout support
        if (timeout_ms > 0) {
            set_nonblocking(sock, true);
        }

        int connect_result = connect(sock, rp->ai_addr, (int)rp->ai_addrlen);

        if (connect_result == 0) {
            // Connected immediately
            break;
        }

#ifdef _WIN32
        if (WSAGetLastError() == WSAEWOULDBLOCK) {
#else
        if (errno == EINPROGRESS) {
#endif
            // Connection in progress, wait for it
            int poll_result = poll_writable(sock, timeout_ms);
            if (poll_result > 0) {
                // Check if connection succeeded
                int so_error;
                socklen_t len = sizeof(so_error);
                if (getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&so_error, &len) == 0 && so_error == 0) {
                    // Success
                    break;
                }
            }
        }

        // Connection failed
        close_socket(sock);
        sock = INVALID_SOCKET_VALUE;
    }

    freeaddrinfo(result);

    if (sock != INVALID_SOCKET_VALUE) {
        // Restore blocking mode and set options
        set_nonblocking(sock, false);
        set_nodelay(sock, true);
        set_keepalive(sock, true);
    }

    return sock;
}

void close_socket(socket_t sock) {
    if (sock != INVALID_SOCKET_VALUE) {
        // Graceful shutdown
        shutdown(sock, 2);  // SHUT_RDWR
        CLOSE_SOCKET(sock);
    }
}

// ============================================================================
// DATA TRANSMISSION
// ============================================================================

bool send_all(socket_t sock, const void* data, size_t size) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t sent = send(sock, (const char*)ptr, (int)remaining, 0);
        if (sent <= 0) {
#ifdef _WIN32
            if (sent == SOCKET_ERROR && WSAGetLastError() == WSAEINTR) continue;
#else
            if (sent == -1 && errno == EINTR) continue;
#endif
            return false;
        }
        ptr += sent;
        remaining -= sent;
    }
    return true;
}

bool recv_all(socket_t sock, void* buffer, size_t size) {
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t received = recv(sock, (char*)ptr, (int)remaining, 0);
        if (received <= 0) {
#ifdef _WIN32
            if (received == SOCKET_ERROR && WSAGetLastError() == WSAEINTR) continue;
#else
            if (received == -1 && errno == EINTR) continue;
#endif
            return false;  // Error or disconnect
        }
        ptr += received;
        remaining -= received;
    }
    return true;
}

ssize_t recv_with_timeout(socket_t sock, void* buffer, size_t size, int timeout_ms) {
    int poll_result = poll_readable(sock, timeout_ms);
    if (poll_result <= 0) {
        return poll_result;  // 0 = timeout, -1 = error
    }

    ssize_t received = recv(sock, (char*)buffer, (int)size, 0);
    return received;
}

int poll_readable(socket_t sock, int timeout_ms) {
#ifdef _WIN32
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(sock, &read_fds);

    struct timeval tv;
    struct timeval* ptv = nullptr;
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ptv = &tv;
    }

    return select(0, &read_fds, nullptr, nullptr, ptv);
#else
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;
    pfd.revents = 0;

    int result = poll(&pfd, 1, timeout_ms);
    if (result > 0 && (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
        return -1;
    }
    return result;
#endif
}

int poll_writable(socket_t sock, int timeout_ms) {
#ifdef _WIN32
    fd_set write_fds;
    FD_ZERO(&write_fds);
    FD_SET(sock, &write_fds);

    struct timeval tv;
    struct timeval* ptv = nullptr;
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ptv = &tv;
    }

    return select(0, nullptr, &write_fds, nullptr, ptv);
#else
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLOUT;
    pfd.revents = 0;

    int result = poll(&pfd, 1, timeout_ms);
    if (result > 0 && (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
        return -1;
    }
    return result;
#endif
}

// ============================================================================
// MESSAGE TRANSMISSION
// ============================================================================

bool send_message(socket_t sock, MessageType type, const void* payload, uint16_t payload_size) {
    MessageHeader header(type, payload_size);

    if (!send_all(sock, &header, sizeof(header))) {
        return false;
    }

    if (payload_size > 0 && payload != nullptr) {
        if (!send_all(sock, payload, payload_size)) {
            return false;
        }
    }

    return true;
}

bool send_message(socket_t sock, MessageType type) {
    return send_message(sock, type, nullptr, 0);
}

bool recv_message_header(socket_t sock, MessageHeader& header, int timeout_ms) {
    if (timeout_ms > 0) {
        int poll_result = poll_readable(sock, timeout_ms);
        if (poll_result <= 0) {
            return false;
        }
    }

    if (!recv_all(sock, &header, sizeof(header))) {
        return false;
    }

    return header.is_valid();
}

bool recv_message_payload(socket_t sock, void* buffer, size_t size) {
    return recv_all(sock, buffer, size);
}

// ============================================================================
// ADDRESS UTILITIES
// ============================================================================

bool parse_address(const std::string& addr, std::string& host, uint16_t& port) {
    // Handle IPv6 addresses in brackets
    size_t bracket_pos = addr.find(']');
    size_t colon_pos;

    if (addr[0] == '[' && bracket_pos != std::string::npos) {
        // IPv6 format: [::1]:port
        host = addr.substr(1, bracket_pos - 1);
        if (bracket_pos + 1 < addr.size() && addr[bracket_pos + 1] == ':') {
            port = static_cast<uint16_t>(std::stoi(addr.substr(bracket_pos + 2)));
        } else {
            port = DEFAULT_SERVER_PORT;
        }
    } else {
        // IPv4 or hostname format: host:port
        colon_pos = addr.rfind(':');
        if (colon_pos != std::string::npos) {
            host = addr.substr(0, colon_pos);
            port = static_cast<uint16_t>(std::stoi(addr.substr(colon_pos + 1)));
        } else {
            host = addr;
            port = DEFAULT_SERVER_PORT;
        }
    }

    return !host.empty();
}

std::string format_address(const std::string& host, uint16_t port) {
    std::ostringstream oss;
    if (host.find(':') != std::string::npos) {
        // IPv6
        oss << "[" << host << "]:" << port;
    } else {
        oss << host << ":" << port;
    }
    return oss.str();
}

bool resolve_hostname(const std::string& hostname, std::string& ip_address) {
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int ret = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
    if (ret != 0) {
        return false;
    }

    char ip_str[INET_ADDRSTRLEN];
    struct sockaddr_in* addr = (struct sockaddr_in*)result->ai_addr;
    inet_ntop(AF_INET, &addr->sin_addr, ip_str, sizeof(ip_str));
    ip_address = ip_str;

    freeaddrinfo(result);
    return true;
}

std::string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return hostname;
    }
    return "unknown";
}

std::string get_local_ip() {
#ifdef _WIN32
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        return "127.0.0.1";
    }

    struct hostent* host = gethostbyname(hostname);
    if (host == nullptr || host->h_addr_list[0] == nullptr) {
        return "127.0.0.1";
    }

    struct in_addr addr;
    memcpy(&addr, host->h_addr_list[0], sizeof(addr));
    return inet_ntoa(addr);
#else
    struct ifaddrs* ifaddr;
    if (getifaddrs(&ifaddr) == -1) {
        return "127.0.0.1";
    }

    std::string result = "127.0.0.1";

    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        if (ifa->ifa_addr->sa_family != AF_INET) continue;

        // Skip loopback
        if (ifa->ifa_flags & IFF_LOOPBACK) continue;

        struct sockaddr_in* addr = (struct sockaddr_in*)ifa->ifa_addr;
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr->sin_addr, ip_str, sizeof(ip_str));

        // Skip link-local addresses
        if (strncmp(ip_str, "169.254.", 8) == 0) continue;

        result = ip_str;
        break;
    }

    freeifaddrs(ifaddr);
    return result;
#endif
}

} // namespace net

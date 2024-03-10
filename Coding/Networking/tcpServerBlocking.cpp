#include <string>
#include <algorithm>
#include <thread>
#include <iostream>
#include <functional>
#include <stack>
#include <queue>
#include <iostream>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib") // Link with Ws2_32.lib

int main() {
  
  WSADATA wsaData;
  SOCKET listeningSocket, clientSocket;
  sockaddr_in serverAddr, clientAddr;
  int clientAddrSize = sizeof(clientAddr);

  // Initialize Winsock
  if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
    std::cerr << "Failed to initialize Winsock." << std::endl;
    return 1;
  }

  // Create a listening socket
  listeningSocket = socket(AF_INET, SOCK_STREAM, 0);
  if (listeningSocket == INVALID_SOCKET) {
    std::cerr << "Socket creation failed with error: " << WSAGetLastError() << std::endl;
    WSACleanup();
    return 1;
  }

  // Bind the socket to an IP address and port
  serverAddr.sin_addr.s_addr = INADDR_ANY; // Use any available address
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(12345); // Specify a port number

  if (bind(listeningSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
    std::cerr << "Bind failed with error: " << WSAGetLastError() << std::endl;
    closesocket(listeningSocket);
    WSACleanup();
    return 1;
  }

  // Listen for incoming connections
  // SOMAXCONN sets the maximum length of the queue for pending connections.
  if (listen(listeningSocket, SOMAXCONN) == SOCKET_ERROR) {
    std::cerr << "Listen failed with error: " << WSAGetLastError() << std::endl;
    closesocket(listeningSocket);
    WSACleanup();
    return 1;
  }

  // Accept a client socket
  // Waits for an incoming connection
  // Waits in blocking mode... execution is blocked until the connection will be accepted
  // We can also use non-blocking mode
  clientSocket = accept(listeningSocket, (sockaddr*)&clientAddr, &clientAddrSize);
  if (clientSocket == INVALID_SOCKET) {
    std::cerr << "Accept failed with error: " << WSAGetLastError() << std::endl;
    closesocket(listeningSocket);
    WSACleanup();
    return 1;
  }

  // At this point, you can communicate with the clientSocket, e.g., send/receive data

  std::cout << "Client connected!" << std::endl;

  // Close client socket
  closesocket(clientSocket);
  // Close the listening socket
  closesocket(listeningSocket);
  // Cleanup Winsock
  WSACleanup();

  return 0;
}
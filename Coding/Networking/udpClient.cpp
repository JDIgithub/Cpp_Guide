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
  SOCKET udpSocket;
  sockaddr_in serverAddr;
  const char* message = "Hello, UDP Server!";

  // Initialize Winsock
  if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
    std::cerr << "Failed to initialize Winsock." << std::endl;
    return 1;
  }

  // Create a UDP socket
  // SOCK_DGRAM specifies UDP datagram socket
  // IPPROTO_UDP is used for UDP protocol
  udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (udpSocket == INVALID_SOCKET) {
    std::cerr << "Socket creation failed with error: " << WSAGetLastError() << std::endl;
    WSACleanup();
    return 1;
  }

  // Setup the server address
  serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1"); // Specify server's IP address
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(12345); // Specify the server's port number

  // Send a message to the server
  if (sendto(udpSocket, message, strlen(message), 0, 
    (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
    std::cerr << "sendto() failed with error: " << WSAGetLastError() << std::endl;
    closesocket(udpSocket);
    WSACleanup();
    return 1;
  }

  std::cout << "Message sent to server!" << std::endl;

  // Close the socket
  closesocket(udpSocket);
  // Cleanup Winsock
  WSACleanup();

  return 0;
}
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
  SOCKET mySocket;
  sockaddr_in server;

  // Initialize Winsock
  // WSAStartup is called to initialize Winsock DLL
  // The MAKEWORD(2,2) parameter specifies version of WinSock 2.2
  if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
    std::cerr << "Failed to initialize Winsock." << std::endl;
    return 1;
  }

  // Create a socket
  // socket() creates new socket
  // AF_INET specifies the IPv4 address family
  // SOCK_STREAM specifies TCP socket
  // Last argument is the protocol. 0 lets system choose the TCP protocol for stream sockets
  mySocket = socket(AF_INET, SOCK_STREAM, 0);
  if (mySocket == INVALID_SOCKET) {
    std::cerr << "Socket creation failed with error: " << WSAGetLastError() << std::endl;
    WSACleanup();
    return 1;
  }

  // Fill in the structure "server" with the address of the server that we want to connect to.
  server.sin_addr.s_addr = inet_addr("127.0.0.1"); // Specify server's IP address
  server.sin_family = AF_INET;
  server.sin_port = htons(80); // Specify server's port number

  // Connect to server
  if (connect(mySocket, (struct sockaddr *)&server, sizeof(server)) < 0) {
    std::cerr << "Connect failed with error: " << WSAGetLastError() << std::endl;
    closesocket(mySocket);
    WSACleanup();
    return 1;
  }

  std::cout << "Connected to server!" << std::endl;
  // Close the socket
  closesocket(mySocket);
  // Cleanup Winsock
  WSACleanup();

  return 0;
}
# Inter-Process Communication (IPC)

## Pipes

### Purpose

- Pipes are used for communication between processes that are closely related
- They are generally used for passing data from one process to another in unidirectional way

### Types

- **Anonymous Pipes** are used for communication between parent and child processes
- **Named Pipes** are used for communication between any processes and are identified by a name in the file system

### Scope

- Pipes are typically limited to a single system
- They are not suitable for network communication



## Sockets

### Purpose

- Sockets provide a way for processes to communicate either within the same system or over a network
- They are more versatile than pipes and are used for network communication
- It is an endpoint of a two-way communication link between two programs running on the network
- It is bound to a port number so that the TCP layer can identify the application that data is destined to

### Types

#### Stream Sockets (TCP)

- TCP for reliable, connection-oriented communication
- Two-way connection-based byte streams
- They ensure that data will be delivered in the order it was sent without duplication

#### Datagram Sockets (UDP)

- UDP (User Datagram Protocol) for connection-less communication
- They are suitable for broadcasting messages over a network but do not guarantee order or delivery

### Scope

- Sockets can be used for both local (inter-process on the same machine) and remote (across the network) communication


### Socket Programming

- Way to enable communication between different processes either on the same machine or across different machines connected by a network

#### Establishing Connection

1. **Creating Socket:**
  
- We need to create socket using "socket()" function 

2. **Bind the Socket:**

- Then we need to bind the socket to an IP address and port using "bind()"

3. **Listen for Connection:**

- Then listen for connection with "listen()"

4. **Accept Connection:**

- Finally we can accept a connection with "accept()"

#### Sending and Receiving Data

- We can use "send()" and "recv()" functions to transmit and receive data
- Ensure to handle network byte order conversions (big endian/little endian) for integer types

#### Closing Connection

- Once communication is complete, close the socket using "close()" or "shutdown()"


#### Considerations

##### Error Handling

- Robust error handling is crucial in network programming
- Always check return values of socket functions and handle errors appropriately
  
##### Security

- Be mindful of security implications
- Encrypt sensitive data and consider using SSL/TLS for secure communication

##### Concurrency

- Handle multiple connections simultaneously
- This might involve multi-threading or asynchronous I/O operations

##### Performance

- Efficient data handling and buffer management are important for high-performance network applications

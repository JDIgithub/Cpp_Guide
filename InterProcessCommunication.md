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

### Types

- **Stream Sockets** using TCP for reliable, connection-oriented communication
- **Datagram Sockets** using UDP for connectionless communication

### Scope

- Sockets can be used for both local (inter-process on the same machine) and remote (across the network) communication
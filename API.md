# API (Application Programming Interface)

## Introduction

- It is set of rules and protocols for building and interacting with software applications
- APIs allow different software systems to communicate with each other
- They are used to enable the integrations to share data and communicate witch each other

### Interface

- API is an interface much like a user interface but instead of being designed for humans to use it is designed for software or applications
- It defines the ways by which software applications can interact with it

### Functionality Sharing

- APIs are used to expose the functionality of a server, application or service to other applications
- For example, a weather application might use an API to gather weather data from various online sources

### Data Exchange

- APIs are often used for data exchange
- They can send and receive data allowing different software systems to communicate

### Automation

- APIs can automate tasks by allowing different software systems to interact without human intervention
- This is common in business environments where different systems need to synchronize data or perform actions automatically

### Types of API

- web APIs
- OS APIs
- database APIs
- and much more
  

### Formats and Protocols

- APIs use specific formats and protocols such as **REST**, **SOAP** and **GraphQL**.
- These define how requests and responses are formatted and transmitted

#### REST (Representational State Transfer)

#### SOAP (Simple Object Access Protocol)

#### GraphQL


### WinApi

- Windows API is Microsoft's core set of application programming interfaces available in the Microsoft Windows operating systems
- It is primarily used for developing desktop applications and interacts directly with the underlying Windows OS to manage UI components, work with files and folders, handle system resources and more


## WinAPI

- Windows API is Microsoft's core set of application programming interfaces available in the Microsoft Windows operating systems
- It is primarily used for developing desktop applications and interacts directly with the underlying Windows OS to manage UI components, work with files and folders, handle system resources and more

### Key Components

1. **User Interface**

- **Windows and Messages**
  - WinApi provides functions to create and manage windows, which are the basic elements of GUI application
  - It also handles message processing
  - Every action (like mouse click) generates messages that are processed by the application

- **Common Controls**
  - These are predefined classes for common UI elements like buttons, text, boxes, lists, etc.
  
2. **GDI (Graphics Device Interface)**
   - Used for representing graphical objects and transmitting them to output devices such as monitors and printers
   - Includes functions for drawing lines, shapes and text as well as for managing fonts and colors
  
3. **File and I/O Operations**
   - Functions for file handling, reading, writing as well as more complex file system operations
   - Also includes APIs for networking and internet operations

4. **Memory and Resource Management**
   - APIs for allocating and managing memory as well as handling other system resources
  
5. **System Services**
   - Functions for accessing and managing system-level resources and settings
   - Includes APIs for threading, process management and inter-process communication

### Programming with WinAPI

- Most commonly used with C and C++
- We need to include various header files in our program to use different components of WinAPI (like **windows.h**)
- **Linking Libraries:** Depending on what API functions we use, we may need to link to specific libraries (like **User32.lib** or **Gdi32.lib**)


#### Best Practices

- **Unicode vs ANSI APIs:** 
  - WinApi offers both Unicode and ANSI versions of functions for dealing with strings
  - It is recommended to use Unicode for better internationalization support
- **Handling Messages:** 
  - Understanding the Windows message loop is crucial for developing GUI applications
- **Resource Management:**
  - Be diligent about resource management
  - Resources like handles and memory allocations should be properly released 

#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>



int countErrorLogs(const std::string& logData) {
  int errorCount = 0;
  size_t pos = 0;

  while (pos < logData.size()) {
    // Find the next occurrence of "ERROR" starting from pos
    size_t errorPos = logData.find("ERROR", pos);

    // If "ERROR" is not found, break the loop
    if (errorPos == std::string::npos) { break; }

     // Increment the error count and update pos to search for the next "ERROR"
    ++errorCount;
    pos = errorPos + 1; // Move past the current "ERROR" to avoid infinite loops
  }
  return errorCount;
}

int main() {
  std::string logData {"[2023-02-25 18:22:30] INFO Backup started\n[2023-02-25 18:23:30] ERROR Disk not found\n[2023-02-25 18:24:00] INFO Attempting retry\n[2023-02-25 18:25:00] ERROR Timeout reached\n[2023-02-25 18:26:30] INFO Backup completed"};
  int result = countErrorLogs(logData);
  std::cout << "The number of ERROR logs is: " << result << std::endl; // Should output 2
  return 0;
}
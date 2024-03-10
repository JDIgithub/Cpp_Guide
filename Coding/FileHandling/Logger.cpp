#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>


int main(int argc, char * argv[]) {
  
  // User should insert argument with minimal severity level
  if(argc != 2) {
    std::cerr << "Usage: log processor <SeverityLevel>" << std::endl;
    return 1;
  }
  
  std::string minSeverity = argv[1];  // User should insert argument with minimal severity level
  std::unordered_map<std::string, int> severityMap { {"INFO",1}, {"DEBUG",2}, {"ERROR",3} };
  
  if (severityMap.find(minSeverity) == severityMap.end()){
    std::cerr << "Invalid severity level. Choose from INFO, DEBUG, ERROR" << std::endl;
    return 1;
  }

  std::ifstream logFile("log.txt");
  if(!logFile.is_open()){
    std::cerr << "Could not open the log.txt file" << std::endl;
  }
  std::string line;

  while(getline(logFile,line)) {

    std::size_t pos = line.find(']');
    if (pos != std::string::npos) {
      std::string severity = line.substr(1, pos - 1); // Removes and stores string after first char '[' till to the last character before ']' 
      if(severityMap[severity] >= severityMap[minSeverity]){
        std::cout << line << std::endl;
      }    
    }



  }

  logFile.close();

  return 0;
}
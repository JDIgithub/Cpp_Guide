#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

std::string compressString(const std::string& str) {
  
  if (str.empty()) return str;

  std::ostringstream compressed;
  char lastChar = str[0];
  int count = 1;

  for (size_t i = 1; i < str.length(); ++i) {
    if (str[i] == lastChar) {   // If current char is same as last one increment counter
      ++count;
    } else {
      compressed << lastChar << count;  // Current character is different -> print the last one with its counter to the stream
      lastChar = str[i];
      count = 1;
    }
  }
    
  // Don't forget to append the last set of characters
  compressed << lastChar << count;

  std::string result = compressed.str();
  return result.length() < str.length() ? result : str;

}

int main() {
    std::string input = "aabcccccaaa";
    std::string output = compressString(input);
    std::cout << "Compressed: " << output << std::endl; // a2b1c5a3
    return 0;
}





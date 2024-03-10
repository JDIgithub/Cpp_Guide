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
#include <stack>
#include <cassert>

#include <iostream>
#include <limits>
#include <cstring>


class Algorithm {

public: 

static int FindMaxSubstringLength(const char* str){
  
  if (str == nullptr || std::strlen(str) == 0) return 0; // Check for empty string or null pointer

  int maxLength = 1; // At least one character will be there in a non-empty string
  int currentLength = 1; // Current sequence length

  // Iterate through the string
  for (int i = 1; str[i] != '\0'; ++i) {
    if (str[i] == str[i - 1]) {
    // Increment the length if the current character is the same as the previous
      ++currentLength;
    } else {
      // Update maxLength if the current sequence is longer
      if (currentLength > maxLength) {
        maxLength = currentLength;
      }
      currentLength = 1; // Reset current length
    }
  }
  // Check one last time in case the longest sequence is at the end of the string
  return currentLength > maxLength ? currentLength : maxLength;
}

};

class Algorithm2 {
public:
  static int FindMaxSubstringLength(const char* str) {
    std::string s(str);
    if (s.empty()) return 0;
    int max_length = 1;
    int current_length = 1;
    char previous_char = s[0];

    std::for_each(std::next(s.begin()), s.end(), 
    [&](char c) {
      if (c == previous_char) {
        ++current_length;
      } else {
        max_length = std::max(max_length, current_length);
        current_length = 1;
      }
      previous_char = c;
    });

    return std::max(max_length, current_length);
  }
};


class Algorithm3 {
public:
  static int FindMaxSubstringLength(const char* str) {
    std::string s(str);
    if (s.empty()) return 0;

    int max_length = 1;
    int current_length = 1;

    for (size_t i = 1; i < s.length(); ++i) {
      if (s[i] == s[i - 1]) {
        ++current_length;
        max_length = std::max(max_length, current_length);
      } else {
        current_length = 1;
      }
    }
    return max_length;
  }
};

int main(){

  int result1 = Algorithm2::FindMaxSubstringLength("abbbcc");
  std::cout << result1 << std::endl;  // 3
  int result2 = Algorithm2::FindMaxSubstringLength("aa");
  std::cout << result2 << std::endl;  // 2

  return 0;

}



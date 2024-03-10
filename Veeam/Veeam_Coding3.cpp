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
#include <climits>

class Convert {
public:
static int StrToInt(const char* str) {

  // Check for null pointer or empty string
  if (str == nullptr || *str == '\0') { return 0; }
  int result = 0;
  int sign = 1;
  int i = 0;

  // Handle leading white spaces
  //while (str[i] == ' ')  { i++; }   // This was mistake

  // Handle sign specifier
  if (str[i] == '-') {
    sign = -1;
    i++;
  }

  // Iterate through the string
  while (str[i] != '\0') {
    // Check for invalid characters
    if (str[i] < '0' || str[i] > '9') { return 0; }

    // Convert character to integer and update result
      result = result * 10 + (str[i] - '0');
      i++;
    }
    return result * sign;
  }
};

class Convert2 {
public:
static int StrToInt(const char* str) {
  // Check for null pointer or empty string
  if (str == nullptr || *str == '\0') { return 0; }
  int result = 0;
  int sign = 1;
  int i = 0;

  // Handle sign specifier
  if (str[i] == '-') {
    sign = -1;
    i++;
  }
  // Iterate through the string
  while (str[i] != '\0') {
    // Check for invalid characters
    if (str[i] < '0' || str[i] > '9') { return 0; }
    // Check for overflow
    if (sign == 1 && (result > INT_MAX / 10 || (result == INT_MAX / 10 && (str[i] - '0') > INT_MAX % 10))) { return 0; }  
    // Check for underflow
    if (sign == -1 && (-result < INT_MIN / 10 || (-result == INT_MIN / 10 && -(str[i] - '0') < INT_MIN % 10))) { return 0; }
    
    result = result * 10 + (str[i] - '0');
    i++;
  }
  return result * sign;
}
};


int main() {

  std::cout << Convert2::StrToInt("100") << std::endl;     // Output: 100
  std::cout << Convert2::StrToInt("   100") << std::endl; // Output: 0
  std::cout << Convert2::StrToInt("123asd") << std::endl; // Output: 0
  std::cout << Convert2::StrToInt("-123") << std::endl;   // Output: -123

  return 0;

}



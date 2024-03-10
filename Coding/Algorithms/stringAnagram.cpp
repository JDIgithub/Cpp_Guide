#include <string>
#include <algorithm>
#include <thread>
#include <iostream>


bool areAnagram(std::string str1, std::string str2) {
    
  if(str1.length() != str2.length()) { return false; }

  std::sort(str1.begin(),str1.end());   // Hybrid sorting algorithm
  std::sort(str2.begin(),str2.end());

  return std::equal(str1.begin(),str1.end(),str2.begin());

}

int main() {

  std::string str1 {"dog"};
  std::string str2 {"god"};

  if(areAnagram(str1,str2)){
    std::cout << "Anagram" << std::endl;
  } else {
    std::cout << "Not anagram" << std::endl;
  }

}
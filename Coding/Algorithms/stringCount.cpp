#include <string>
#include <algorithm>
#include <thread>
#include <iostream>

int countWords (const std::string& str){
  int num {0};
  char prev{' '};
  for(char c : str){
    if(c != ' ' && prev == ' ') { num++; }
    prev = c;
  }
  return num;
}

bool isVowel(char c){
  c = toupper(c);
  if (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' ) {
    return true;
  } else {
    return false;
  }
}

int countVowels(const std::string& str) {

  int count {0};
  for (const char& c: str){
    if(isVowel(c)){ count++;}
  }
  return count;
}

int main()
{
  std::string str {"abc de tak to teda ne"};
  std::cout << countVowels(str) << std::endl;
  std::cout << countWords(str) << std::endl;
}
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <atomic>

using namespace std;

// 58. Length of Last Word
/*

Given a string s consisting of words and spaces, return the length of the last word in the string.
A word is a maximal substring consisting of non-space characters only.

 

Example 1:

  Input: s = "Hello World"
  Output: 5
  Explanation: The last word is "World" with length 5.

Example 2:

  Input: s = "   fly me   to   the moon  "
  Output: 4
  Explanation: The last word is "moon" with length 4.

Example 3:

  Input: s = "luffy is still joyboy"
  Output: 6
  Explanation: The last word is "joyboy" with length 6.
 

Constraints:

1 <= s.length <= 104
s consists of only English letters and spaces ' '.
There will be at least one word in s.
  
*/



int lengthOfLastWord(std::string str) {
  int lengthLast = 0;
  for(int i = str.size()-1; i >= 0; i--){
    if(str[i] != ' ') {
      lengthLast++; 
    } else {
      if(lengthLast > 0){
        break;
      }
    }
  }
  return lengthLast;
}


int main(){

  std::string str = {"Hello World"};

  std::cout << lengthOfLastWord(str);


  return 0;
}







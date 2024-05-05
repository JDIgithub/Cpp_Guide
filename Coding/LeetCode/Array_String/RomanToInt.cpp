#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <math.h>
using namespace std;

// 13. Roman to Integer

/*

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000


Roman numerals are usually written largest to smallest from left to right. 
However, the numeral for four is not IIII. Instead, the number four is written as IV. 
Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX.
There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.


Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
 

Constraints:

1 <= s.length <= 15
s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
It is guaranteed that s is a valid roman numeral in the range [1, 3999].

*/

int letterToNum(char c){

  int num = 0;
  switch(c){

    case 'I': 
      num = 1;
      break;
    case 'V': 
      num = 5;
      break;
    case 'X': 
      num = 10;
      break;
    case 'L': 
      num = 50;
      break;
    case 'C': 
      num = 100;
      break;
    case 'D': 
      num = 500;
      break;   
    case 'M': 
      num = 1000;
      break;
    default:
      break;
  }


  return num;
}

int romanToInt(string s) {

  if(s.empty()) return 0;
  int integer = letterToNum(s[0]);
  int prev = integer;

  for(int i = 1; i < s.length(); i++){

    if(letterToNum(s[i]) <= prev){
      integer += letterToNum(s[i]);
    } else {
      integer = integer + letterToNum(s[i]) - 2*prev;
    }

    prev = letterToNum(s[i]);
  }

  return  integer;
}


int main(){

  std::vector<int> nums {1,8,6,2,5,25,8,25,7};
  std::cout << romanToInt("III");
  



  return 0;
}



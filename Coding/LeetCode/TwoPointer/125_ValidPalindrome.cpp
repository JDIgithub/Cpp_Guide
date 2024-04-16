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

// 125. Valid Palindrome
/*

A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. 
Alphanumeric characters include letters and numbers.
Given a string s, return true if it is a palindrome, or false otherwise.


Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.
Example 3:

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.

*/
// Two pointers for the character comparsion
// Time: O(n)  Space O(1)
bool isPalindrome(std::string s) {

  if (s.empty()) return true;
  int pointerA = 0;
  int pointerB = s.length()-1;

  while(pointerA <= pointerB){

    if(isalnum(s[pointerA])){                 // If character is aplhanumerical, lower it and send it to compare
      s[pointerA] = tolower(s[pointerA]);
    }else{                                    // else try another character
      pointerA++;
      continue;
    }

    if(isalnum(s[pointerB])){                 // If character is aplhanumerical, lower it and send it to compare
      s[pointerB] = tolower(s[pointerB]);
    }else{                                    // else try anoter character
      pointerB--;
      continue;
    }

    char a= s[pointerA];
    char b = s[pointerB];
    if(s[pointerA] != s[pointerB]){
      return false;
    }

    pointerA++;
    pointerB--;

  }

  return true;

}


int main(){

  std::string s {"A man, a plan, a canal: Panama"};
  std::cout << isPalindrome(s);



  return 0;
}


